"""
live_localize.py — LIVE hazard detection + distance measurement on video frames.

Input : a video file, a webcam, a single image, or a folder of images (treated as frames).
Output: the same frames, annotated with each hazard's distance, direction, size and uncertainty,
        plus a small top-down map in the corner. Saved as a video (and shown live with --show).

Run from the repo root:
    python scripts/live_localize.py --source data/sample_images --show        # folder of images
    python scripts/live_localize.py --source rover_drive.mp4 --show           # video file
    python scripts/live_localize.py --source 0 --show                         # webcam
Options:
    --h 0.5 --pitch 25 --hfov 90      override the camera mount from config/mission.json
    --horizon-row 120                 compute the pitch from the visible horizon row instead
    --every 2                         run the detector every 2nd frame (re-use boxes in between)
    --show-small / --hide-small       override hazard_filter.show_small from the config
Keys (with --show): q / Esc = quit, space = pause.

Method (see docs): pinhole camera -> ray through the hazard's foot pixel -> intersection with the
flat ground -> range, bearing, width, height and a 1-sigma uncertainty from the error model.
Hazard filter (config "hazard_filter"): a detection is a HAZARD only if its metric size exceeds
what the rover can drive over / cross. Smaller detections are "small": drawn faintly (or hidden),
never counted, never put on the map, and should never be transmitted.
"""
import argparse
import glob
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from crater.config import build_error_model, load_config, model_path  # noqa: E402
from crater.localization import (GroundProjector, Intrinsics, Mount, Pose, covariance,  # noqa: E402
                                 range_cross_sigmas)
from crater.perception import HazardDetector  # noqa: E402

COLORS = {"rock": (0, 0, 255), "crater": (0, 140, 255)}
LOW_CONF = (150, 150, 150)                                     # grey = beyond the trusted zone
SMALL = (110, 150, 110)                                        # dim green = too small to be a hazard

HAZARD_DEFAULTS = {
    "rock_min_height_m": 0.1,
    "rock_min_width_m": 0.2,
    "crater_min_width_m": 0.3,
    "show_small": True,
    "rock_rule": "any",        # "any": tall OR wide enough (safe side) | "all": tall AND wide
}


# ------------------------------------------------------------------------------------ hazard filter
def is_hazard(cls, width, height, hf):
    """True if a localized detection is big enough to threaten the rover.
    Detections with no metric size (beyond range) cannot be judged: they are kept, never dropped blindly."""
    if width is None:
        return True
    if cls == "crater":
        return width >= hf["crater_min_width_m"]
    if cls == "rock":
        wide = width >= hf["rock_min_width_m"]
        tall = height is not None and height >= hf["rock_min_height_m"]
        if height is None:                                   # no height estimate: judge on width only
            return wide
        return (wide and tall) if hf["rock_rule"] == "all" else (wide or tall)
    return True                                              # unknown class: keep it


# ------------------------------------------------------------------------------------ input
def open_source(src):
    """Yields (index, frame, fps). Accepts webcam index, video file, image file or image folder."""
    if src.isdigit():
        cap = cv2.VideoCapture(int(src))
    elif Path(src).is_dir():
        files = sorted(p for ext in ("*.jpg", "*.jpeg", "*.png") for p in glob.glob(str(Path(src) / ext)))
        for i, f in enumerate(files):
            frame = cv2.imread(f)
            if frame is not None:
                yield i, frame, 2.0                             # image folder: 2 frames per second
        return
    elif Path(src).suffix.lower() in (".jpg", ".jpeg", ".png"):
        frame = cv2.imread(src)
        if frame is not None:
            yield 0, frame, 1.0
        return
    else:
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"Cannot open source: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield i, frame, fps
        i += 1
    cap.release()


# ------------------------------------------------------------------------------------ drawing
def _label_size(lines, font=cv2.FONT_HERSHEY_SIMPLEX, fs=0.45, th=1):
    sizes = [cv2.getTextSize(t, font, fs, th)[0] for t in lines]
    return max(s[0] for s in sizes) + 8, sum(s[1] + 6 for s in sizes) + 4, sizes


def _overlaps(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def put_label(img, lines, candidates, color, taken):
    """Draw a text block with a dark background at the first candidate (x, y_top) that stays
    inside the image and does not overlap any rectangle in `taken`. Returns its rectangle."""
    font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
    w, h, sizes = _label_size(lines)
    H, W = img.shape[:2]
    rect = None
    for x, y in candidates:
        x, y = int(min(max(x, 0), W - w)), int(min(max(y, 0), H - h))
        r = (x, y, x + w, y + h)
        if not any(_overlaps(r, t) for t in taken):
            rect = r
            break
    if rect is None:                                     # no free spot: use the first candidate
        x, y = candidates[0]
        x, y = int(min(max(x, 0), W - w)), int(min(max(y, 0), H - h))
        rect = (x, y, x + w, y + h)
    x0, y0, x1, y1 = rect
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
    cy = y0 + 4
    for t, sz in zip(lines, sizes):
        cy += sz[1] + 6
        cv2.putText(img, t, (x0 + 4, cy - 4), font, fs, color, th, cv2.LINE_AA)
    taken.append(rect)
    return rect


def mini_map(results, size=220, range_m=8.0):
    m = np.full((size, size, 3), 20, np.uint8)
    s = (size - 20) / range_m
    rover = (size // 2, size - 10)
    for r_m in (2, 5, 8):
        cv2.circle(m, rover, int(r_m * s), (0, 140, 0) if r_m == 5 else (70, 70, 70), 1)
        cv2.putText(m, f"{r_m}m", (rover[0] + 3, rover[1] - int(r_m * s) + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)
    cv2.rectangle(m, (rover[0] - 5, rover[1] - 4), (rover[0] + 5, rover[1] + 4), (255, 255, 255), -1)
    for r in results:
        p = (int(rover[0] - r["left"] * s), int(rover[1] - r["fwd"] * s))
        c = r["color"]
        cv2.circle(m, p, max(2, int(2 * r["sigma"] * s)), c, 1)    # 2-sigma along-range circle
        cv2.circle(m, p, 3, c, -1)
    cv2.putText(m, "top view", (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    return m


def annotate(frame, results, horizon_row, fps_txt):
    out = frame.copy()
    H, W = out.shape[:2]
    taken = []
    hazards = [r for r in results if r["hazard"]]
    small = [r for r in results if not r["hazard"]]
    # mini map in the top-right corner: real hazards only
    mm = mini_map([r for r in hazards if r["range"] is not None], size=min(180, W // 3, H // 3))
    mh, mw = mm.shape[:2]
    out[10:10 + mh, W - mw - 10:W - 10] = mm
    taken.append((W - mw - 10, 10, W - 10, 10 + mh))
    if 0 <= horizon_row < H:
        cv2.line(out, (0, int(horizon_row)), (W - mw - 14, int(horizon_row)), (0, 255, 255), 1)
    nearest = min((r["range"] for r in hazards if r["range"] is not None), default=None)
    status = f"{len(hazards)} hazards" + (f" | nearest {nearest:.2f} m" if nearest is not None else "")
    if small:
        status += f" | {len(small)} small ignored"
    put_label(out, [status, fps_txt], [(10, 10)], (255, 255, 255), taken)

    for r in results:                                    # boxes first, so labels never hide a box edge
        x1, y1, x2, y2 = r["box"]
        cv2.rectangle(out, (x1, y1), (x2, y2), r["color"], 2 if r["hazard"] else 1)
        if r["hazard"]:
            cv2.circle(out, tuple(r["foot_px"]), 4, (255, 255, 0), -1)
        taken.append((x1, y1, x2 + 1, y2 + 6))           # labels must not cover any box or foot dot
    for r in sorted(results, key=lambda q: q["box"][3], reverse=True):   # nearest hazards labelled first
        x1, y1, x2, y2 = r["box"]
        if not r["hazard"]:                              # small: one short line, dim colour
            lines, color = [f"small {r['cls']} {r['width']:.2f} m"], SMALL
        elif r["range"] is None:
            lines, color = [f"{r['cls'].upper()} {r['conf']:.2f}", "beyond range"], LOW_CONF
        else:
            side = "L" if r["bearing"] >= 0 else "R"
            lines = [f"{r['cls'].upper()} {r['conf']:.2f}{'' if r['trusted'] else ' (low conf.)'}",
                     f"{r['range']:.2f} m +/-{r['sigma']:.2f}  {abs(r['bearing']):.0f} deg {side}"]
            if r["width"] is not None:
                size_txt = f"w {r['width']:.2f} m"
                if r["cls"] == "rock" and r["height"] is not None:
                    size_txt += f"  h {r['height']:.2f} m"
                lines.append(size_txt)
            color = r["color"]
        w, h, _ = _label_size(lines)
        cands = [(x1, y1 - h - 2), (x1, y2 + 6), (x2 + 6, y1), (x1 - w - 6, y1)]   # above, below, right, left
        for k in range(1, 6):                                                # then stacked further away
            cands += [(x1, y1 - h - 2 - k * (h + 2)), (x1, y2 + 6 + k * (h + 2))]
        put_label(out, lines, cands, color, taken)
    return out


# ------------------------------------------------------------------------------------ main
def main():
    cfg = load_config()
    hf = {**HAZARD_DEFAULTS, **cfg.get("hazard_filter", {})}
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="video file, webcam index, image, or image folder")
    ap.add_argument("--out", default=str(ROOT / "test_outputs" / "live" / "annotated.mp4"))
    ap.add_argument("--show", action="store_true", help="display frames live")
    ap.add_argument("--every", type=int, default=1, help="run the detector every N frames")
    ap.add_argument("--h", type=float, default=cfg["mount"]["h"])
    ap.add_argument("--pitch", type=float, default=cfg["mount"]["pitch_deg"])
    ap.add_argument("--hfov", type=float, default=cfg["camera"]["hfov_deg"])
    ap.add_argument("--horizon-row", type=float, default=None)
    ap.add_argument("--show-small", action=argparse.BooleanOptionalAction, default=hf["show_small"],
                    help="draw detections below the hazard thresholds (dimmed)")
    args = ap.parse_args()

    mp = model_path(cfg)
    if not mp.exists():
        sys.exit(f"Model not found: {mp}")
    d = cfg["detector"]
    det = HazardDetector(str(mp), conf_thres=d["conf_thres"], iou_thres=d["iou_thres"], threads=d["threads"])
    err, trusted_m = build_error_model(cfg), cfg["localization"]["trusted_range_m"]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = out_path.parent / "frames"
    frames_dir.mkdir(exist_ok=True)

    writer, proj, size, detections, times = None, None, None, [], []
    raw_total, hazard_total = 0, 0
    for i, frame, fps in open_source(args.source):
        H, W = frame.shape[:2]
        if (W, H) != size:                                       # (re)build the camera model for this size
            size = (W, H)
            K = Intrinsics.from_fov(W, H, args.hfov)
            pitch = args.pitch if args.horizon_row is None else math.degrees(math.atan((K.cy - args.horizon_row) / K.fy))
            proj = GroundProjector(K, Mount(tx=cfg["mount"]["tx"], ty=cfg["mount"]["ty"], h=args.h, pitch_deg=pitch),
                                   cfg["localization"]["min_depression_deg"])
            if writer is not None:
                writer.release()
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

        t0 = time.perf_counter()
        if i % args.every == 0:
            detections = det.detect(frame)                      # PERCEIVE
        results = []
        for hz in detections:                                    # LOCALIZE each detection
            r = {"cls": hz["cls"], "conf": hz["conf"], "box": hz["box"], "foot_px": hz["foot_px"],
                 "color": COLORS.get(hz["cls"], (0, 255, 0)), "range": None,
                 "width": None, "height": None, "hazard": True}
            g = proj.hazard_geometry(hz["box"], Pose())
            if g is not None:
                cov = covariance(proj, *hz["foot_px"], Pose(), err)
                sigma = range_cross_sigmas(cov[1], Pose(), cov[0])[0] if cov else float("nan")
                trusted = g["range_m"] <= trusted_m
                r.update(range=g["range_m"], bearing=g["bearing_deg"], fwd=g["E"], left=g["N"],
                         width=g["width_m"], height=g["height_m"], sigma=sigma, trusted=trusted)
                if not trusted:
                    r["color"] = LOW_CONF
            r["hazard"] = is_hazard(r["cls"], r["width"], r["height"], hf)   # FILTER by physical size
            results.append(r)
        raw_total += len(results)
        hazard_total += sum(r["hazard"] for r in results)
        shown = [r for r in results if r["hazard"] or args.show_small]
        ms = (time.perf_counter() - t0) * 1000
        times.append(ms)

        out = annotate(frame, shown, proj.horizon_row(), f"{ms:.0f} ms/frame (detect + localize + filter)")
        writer.write(out)
        cv2.imwrite(str(frames_dir / f"frame_{i:05d}.jpg"), out)
        hz_txt = ", ".join(f"{r['cls']} {r['range']:.2f} m" for r in results if r["hazard"] and r["range"] is not None)
        n_small = sum(not r["hazard"] for r in results)
        print(f"frame {i}: {hz_txt or 'no hazard in range'}  [{n_small} small ignored, {ms:.0f} ms]")
        if args.show:
            cv2.imshow("C.R.A.T.E.R live localization", out)
            k = cv2.waitKey(1 if fps > 5 else 300) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord(" "):
                cv2.waitKey(0)

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    if times:
        print(f"\n{len(times)} frames, average {sum(times) / len(times):.0f} ms per frame. "
              f"Detections: {raw_total} raw -> {hazard_total} hazards "
              f"({raw_total - hazard_total} filtered as too small). "
              f"Video: {out_path}  Frames: {frames_dir}")


if __name__ == "__main__":
    main()