"""
run_demo.py — ONE command: video -> perceive -> localize -> hazard map -> binary packets
              -> lossy link -> ground station -> web dashboard.

    python scripts/run_demo.py --source video.mp4
    python scripts/run_demo.py --source video.mp4 --loop --loss 0.3 --horizon-row 180
Then open http://127.0.0.1:8000

The rover runs in real time: frames the detector cannot keep up with are skipped (like real flight
software, it never builds a backlog). Rover motion is a SIMULATED straight drive east at --speed m/s
(the video provides the camera view, not the odometry). Ctrl-C to stop.
"""
import argparse
import json
import math
import socket
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from crater.config import build_error_model, load_config, model_path  # noqa: E402
from crater.ground import GroundStation, VideoFeed, serve  # noqa: E402
from crater.hazard_map import HAZARD_DEFAULTS, HazardMap, is_hazard  # noqa: E402
from crater.link import LINK_DEFAULTS, LinkEmulator  # noqa: E402
from crater.localization import (GroundProjector, Intrinsics, Mount, Pose, covariance,  # noqa: E402
                                 ellipse)
from crater.perception import HazardDetector  # noqa: E402
from crater.telemetry import RoverState, encode_packet, max_hazards_for  # noqa: E402

LINK_PORT, GROUND_PORT = 9101, 9102


def frame_gen(src, loop):
    while True:
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            sys.exit(f"Cannot open source: {src} (wrong path, or codec: re-encode to H.264 with ffmpeg)")
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            yield fr
        cap.release()
        if not loop:
            return


def main():
    cfg = load_config()
    hf = {**HAZARD_DEFAULTS, **cfg.get("hazard_filter", {})}
    lk = {**LINK_DEFAULTS, **cfg.get("link", {})}
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="video file")
    ap.add_argument("--loop", action=argparse.BooleanOptionalAction, default=True, help="restart the video when it ends")
    ap.add_argument("--speed", type=float, default=0.1, help="simulated rover speed, m/s")
    ap.add_argument("--port", type=int, default=8000, help="dashboard port")
    ap.add_argument("--every", type=int, default=1, help="run the detector every N frames")
    ap.add_argument("--tx-period", type=float, default=1.0, help="seconds between packets")
    ap.add_argument("--h", type=float, default=cfg["mount"]["h"])
    ap.add_argument("--pitch", type=float, default=cfg["mount"]["pitch_deg"])
    ap.add_argument("--hfov", type=float, default=cfg["camera"]["hfov_deg"])
    ap.add_argument("--horizon-row", type=float, default=None)
    ap.add_argument("--loss", type=float, default=lk["loss"], help="packet loss 0-1")
    ap.add_argument("--delay-ms", type=float, default=lk["delay_ms"])
    ap.add_argument("--jitter-ms", type=float, default=lk["jitter_ms"])
    ap.add_argument("--bandwidth-bps", type=float, default=lk["bandwidth_bps"])
    ap.add_argument("--corrupt", type=float, default=lk["corrupt"], help="probability of a bit flip per packet")
    ap.add_argument("--headless", action="store_true", help="no dashboard server: run for --duration, write a report")
    ap.add_argument("--duration", type=float, default=60.0, help="headless run length, seconds")
    ap.add_argument("--report", default=None, help="write a JSON report of the run to this path")
    args = ap.parse_args()

    mp = model_path(cfg)
    if not mp.exists():
        sys.exit(f"Model not found: {mp}")
    d = cfg["detector"]
    det = HazardDetector(str(mp), conf_thres=d["conf_thres"], iou_thres=d["iou_thres"], threads=d["threads"])
    err, trusted_m = build_error_model(cfg), cfg["localization"]["trusted_range_m"]
    pose_sigma = cfg["error_model"]["sigma_pose_xy"]
    rover_id = cfg["telemetry"]["rover_id"]
    max_h = max_hazards_for(cfg["telemetry"]["max_packet_bytes"])

    link = LinkEmulator(("127.0.0.1", LINK_PORT), ("127.0.0.1", GROUND_PORT), loss=args.loss,
                        delay_ms=args.delay_ms, jitter_ms=args.jitter_ms, bandwidth_bps=args.bandwidth_bps,
                        queue_max=lk["queue_max"], corrupt=args.corrupt, burst=lk["burst"])
    ground = GroundStation(GROUND_PORT, link)
    link.start()
    ground.start()
    feed = VideoFeed()
    if not args.headless:
        serve(ground, link, args.port, feed)
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"Headless run, {args.duration:.0f} s" if args.headless else
          f"Mission Control: http://127.0.0.1:{args.port}   (Ctrl-C to stop)")
    print(f"Link: loss {args.loss:.0%}, delay {args.delay_ms:.0f}+-{args.jitter_ms:.0f} ms, "
          f"{args.bandwidth_bps:.0f} bit/s | packets <= {cfg['telemetry']['max_packet_bytes']} B ({max_h} hazards)")

    probe = cv2.VideoCapture(args.source)
    fps = probe.get(cv2.CAP_PROP_FPS) or 15.0
    probe.release()

    hmap, frames = HazardMap(), frame_gen(args.source, args.loop)
    size, proj, detections = None, None, []
    consumed, seq, next_tx, t0 = 0, 0, 0.0, time.monotonic()
    try:
        while True:
            t = time.monotonic() - t0
            if args.headless and t >= args.duration:
                break
            try:
                for _ in range(max(0, int(t * fps) - consumed)):      # behind real time: skip frames
                    next(frames)
                    consumed += 1
                wait = consumed / fps - t
                if wait > 0:
                    time.sleep(wait)
                frame = next(frames)
            except StopIteration:
                if args.headless:
                    break
                print("Video ended. Dashboard stays up (data keeps ageing). Ctrl-C to stop.")
                while True:
                    time.sleep(1)
            consumed += 1
            t = time.monotonic() - t0
            H, W = frame.shape[:2]
            if (W, H) != size:                                         # build the camera model for this size
                size = (W, H)
                K = Intrinsics.from_fov(W, H, args.hfov)
                pitch = args.pitch if args.horizon_row is None else math.degrees(math.atan((K.cy - args.horizon_row) / K.fy))
                proj = GroundProjector(K, Mount(tx=cfg["mount"]["tx"], ty=cfg["mount"]["ty"], h=args.h, pitch_deg=pitch),
                                       cfg["localization"]["min_depression_deg"])

            rE, rN, yaw = args.speed * t, 0.0, 0.0                     # SIMULATED pose: straight drive east
            if consumed % args.every == 0:
                detections = det.detect(frame)                         # PERCEIVE
            obs, drawn = [], []
            for hz in detections:
                g = proj.hazard_geometry(hz["box"], Pose())            # LOCALIZE (rover frame)
                if g is None:
                    drawn.append((hz["box"], (150, 150, 150), "beyond range"))
                    continue                                           # not localizable: not sent
                if not is_hazard(hz["cls"], g["width_m"], g["height_m"], hf, distance_m=g["range_m"]):
                    drawn.append((hz["box"], (110, 150, 110), "small"))
                    continue                                           # too small: not sent
                if g["range_m"] > trusted_m:
                    drawn.append((hz["box"], (150, 150, 150), f"{hz['cls']} {g['range_m']:.1f} m (far)"))
                    continue                                           # beyond trusted range: not sent
                drawn.append((hz["box"], (0, 0, 255) if hz["cls"] == "rock" else (0, 140, 255),
                              f"{hz['cls']} {g['range_m']:.1f} m"))
                cov = covariance(proj, *hz["foot_px"], Pose(), err)
                unc = ellipse(cov[1])[0] if cov else 2.0
                unc = math.hypot(unc, 2 * pose_sigma)                  # add rover pose uncertainty
                c, s_ = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
                fwd, left = g["E"], g["N"]
                obs.append({"cls": hz["cls"], "E": rE + fwd * c - left * s_, "N": rN + fwd * s_ + left * c,
                            "unc95": unc, "conf": hz["conf"] * (1.0 if g["range_m"] <= trusted_m else 0.5),
                            "width": g["width_m"] or 0.0})
            hmap.update(obs, t)

            scale = max(1.0, W / 640.0)
            thick = max(1, int(round(2 * scale)))
            fs = max(0.4, 0.45 * scale)
            vis = frame.copy()
            for (x1, y1, x2, y2), col, label in drawn:
                cv2.rectangle(vis, (x1, y1), (x2, y2), col, thick)
                cv2.putText(vis, label, (x1, max(int(18 * scale), y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, fs, col, max(1, int(scale)), cv2.LINE_AA)
            if W > 640:
                vis = cv2.resize(vis, (640, int(H * 640 / W)))
            okj, jpg = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if okj:
                feed.set(jpg.tobytes())

            if t >= next_tx:                                           # TRANSMIT
                hzs = hmap.packet_hazards(rE, rN, t, max_h)
                pkt = encode_packet(seq, rover_id, RoverState(t=t, E=rE, N=rN, yaw_deg=yaw, pose_sigma=pose_sigma), hzs)
                tx.sendto(pkt, ("127.0.0.1", LINK_PORT))
                print(f"t={t:6.1f}s  seq {seq:4d}  {len(pkt):3d} B  {len(hzs)} hazards "
                      f"({len(hmap.tracks)} tracked)  rover E={rE:.1f} m")
                seq, next_tx = seq + 1, next_tx + args.tx_period
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if args.report:
            time.sleep(args.delay_ms / 1000 * 2 + 0.5)                 # let in-flight packets arrive
            snap, lks = ground.snapshot(), link.snapshot()
            rep = {"source": args.source, "duration_s": round(time.monotonic() - t0, 1), "frames_consumed": consumed,
                   "packets_sent": seq, "hazards_tracked": len(hmap.tracks),
                   "hazards_confirmed": sum(tr.hits >= hmap.confirm_hits for tr in hmap.tracks.values()),
                   "hazards_at_ground": len(snap["hazards"]),
                   "link": {k: lks[k] for k in ("sent", "delivered", "dropped_loss", "dropped_blackout", "dropped_queue",
                                                "corrupted", "bytes_in", "bytes_out")},
                   "ground": {k: snap["link"][k] for k in ("status", "loss_pct", "rx", "corrupted", "bytes_rx", "latency_ms")}}
            Path(args.report).parent.mkdir(parents=True, exist_ok=True)
            Path(args.report).write_text(json.dumps(rep, indent=2))
            print(f"Report written: {args.report}")
        link.stop()
        ground.stop()


if __name__ == "__main__":
    main()
