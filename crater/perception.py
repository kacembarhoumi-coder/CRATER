"""
rover_perception.py — onboard hazard detection for the C.R.A.T.E.R rover.

Dependencies: numpy, opencv-python, onnxruntime  (NO PyTorch, NO internet).

Usage:
    det = HazardDetector("crater_v1.onnx")
    hazards = det.detect(frame_bgr)
    # -> [{"cls": "rock", "conf": 0.91, "box": [x1, y1, x2, y2], "foot_px": [u, v]}, ...]
"""
import time
import cv2
import numpy as np
import onnxruntime as ort

CLASS_NAMES = ["rock", "crater"]

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def preprocess(img_bgr):
    """EXACTLY the same preprocessing used to build the training set."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = _clahe.apply(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def letterbox(img, size):
    """Resize keeping the aspect ratio, pad the rest with gray (like YOLO does)."""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return canvas, scale, pad_x, pad_y


class HazardDetector:
    def __init__(self, onnx_path, conf_thres=0.35, iou_thres=0.5, threads=1, class_names=CLASS_NAMES):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads      # emulate a weak, single-core rover CPU
        opts.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        self.size = inp.shape[2] if isinstance(inp.shape[2], int) else 320
        self.conf_thres, self.iou_thres = conf_thres, iou_thres
        self.class_names = class_names
        self.last_ms = 0.0

    def detect(self, frame_bgr):
        t0 = time.perf_counter()
        H, W = frame_bgr.shape[:2]
        img, scale, pad_x, pad_y = letterbox(preprocess(frame_bgr), self.size)
        blob = img[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0   # BGR->RGB, HWC->CHW

        out = self.sess.run(None, {self.input_name: blob})[0][0]   # shape (4 + n_classes, n_candidates)
        out = out.T                                                # -> (n_candidates, 4 + n_classes)
        scores_all = out[:, 4:]
        cls_ids = scores_all.argmax(axis=1)
        scores = scores_all[np.arange(len(out)), cls_ids]
        keep = scores >= self.conf_thres
        out, cls_ids, scores = out[keep], cls_ids[keep], scores[keep]

        hazards = []
        if len(out):
            cx, cy, bw, bh = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
            # undo the letterbox: model pixels -> original image pixels
            x1 = np.clip((cx - bw / 2 - pad_x) / scale, 0, W - 1)
            y1 = np.clip((cy - bh / 2 - pad_y) / scale, 0, H - 1)
            x2 = np.clip((cx + bw / 2 - pad_x) / scale, 0, W - 1)
            y2 = np.clip((cy + bh / 2 - pad_y) / scale, 0, H - 1)
            rects = [[float(a), float(b), float(c - a), float(d - b)] for a, b, c, d in zip(x1, y1, x2, y2)]
            idx = cv2.dnn.NMSBoxes(rects, scores.astype(float).tolist(), self.conf_thres, self.iou_thres)
            for i in np.array(idx).flatten():
                hazards.append({
                    "cls": self.class_names[int(cls_ids[i])],
                    "conf": round(float(scores[i]), 3),
                    "box": [int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])],
                    # bottom-center of the box = where the hazard touches the ground.
                    # This is the pixel the Localization step will turn into coordinates.
                    "foot_px": [int((x1[i] + x2[i]) / 2), int(y2[i])],
                })
        self.last_ms = (time.perf_counter() - t0) * 1000
        return hazards


def draw_hazards(frame_bgr, hazards):
    img = frame_bgr.copy()
    colors = {"rock": (0, 0, 255), "crater": (0, 140, 255)}
    for h in hazards:
        x1, y1, x2, y2 = h["box"]
        c = colors.get(h["cls"], (0, 255, 0))
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
        cv2.circle(img, tuple(h["foot_px"]), 4, (255, 255, 0), -1)
        cv2.putText(img, f'{h["cls"]} {h["conf"]:.2f}', (x1, max(y1 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
    return img
