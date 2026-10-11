# C.R.A.T.E.R — Constrained Reconnaissance, Autonomy & Telemetry in Extreme Regions

ATEC'26 · A 100% software-emulated lunar scout: it detects surface hazards on board, turns them into
coordinates relative to the landing site, and sends them to Mission Control over a degraded radio link
in packets of a few bytes.

```
camera frames ──► PERCEIVE ──► LOCALIZE ──► TRANSMIT ──► (lossy link) ──► MISSION CONTROL
                  YOLO11n       pinhole +     binary        packet loss,      dashboard: map,
                  ONNX, 1 CPU   ground plane  packets, CRC  blackouts         data age, uncertainty
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/run_checks.py         # runs all tests and writes test_outputs/
```

The rover side needs only `numpy`, `opencv-python` and `onnxruntime`: no PyTorch, no GPU, no internet.

## Repository layout

| Path | Contents |
|---|---|
| `crater/perception.py` | Onboard hazard detector: grayscale + CLAHE preprocessing, YOLO11n ONNX inference, NMS, `foot_px` |
| `crater/localization.py` | Pixel → landing-site coordinates (pinhole, ground plane, rover pose) and error propagation |
| `crater/telemetry.py` | Binary packet encoder/decoder with CRC-16 |
| `crater/config.py` | Loads `config/mission.json`, the single source of all mission parameters |
| `config/mission.json` | Camera, mount, error model, detector and telemetry settings |
| `models/` | `crater_detector.onnx` (deployed model) |
| `tests/` | Unit tests (pytest) |
| `scripts/run_checks.py` | Produces all automated test outputs in `test_outputs/` |
| `notebooks/` | Training notebook (Colab) |
| `docs/` | Localization math and error budget, training results, experiment log |
| `data/sample_images/` | A few raw lunar test images for the perception benchmark |

## Results

### Perception (YOLO11n, 320 px, grayscale + CLAHE)

| Metric | Value |
|---|---|
| Parameters / compute | 2.6 M / 1.6 GFLOPs per frame (4× less than the 640 px default) |
| Speed, ONNX Runtime, **1 CPU thread** | 37.7 ms/frame ≈ 27 FPS (Colab Xeon 2.0 GHz), +14 MB RAM |
| mAP50, full test set (deployed ONNX) | 0.49 |
| mAP50 on **real** Apollo images | rock 0.48 · crater 0.38 |
| Hazard detected as *any* hazard (conf 0.25) | rocks 88% · craters 93% |

At ~0.1 m/s the rover needs about 2 detections per second, i.e. ~75 ms of CPU per second (~7.5% of one
core): it would still fit on a processor ten times slower.

### Localization (640×480, HFOV 90°, camera 0.5 m high, 25° down)

| Distance from camera | Along-range 1σ | Cross-range 1σ |
|---|---|---|
| 2 m | 0.11 m | 0.06 m |
| 5 m | 0.56 m | 0.11 m |
| 8 m | 1.40 m | 0.16 m |

Camera pitch dominates the error; range error grows with distance squared. Hazards beyond 5 m are sent
as low-confidence, rays less than 3° below the horizon are rejected. Full derivation: `docs/`.

### Telemetry

| Hazards per packet | Binary | JSON equivalent |
|---|---|---|
| 0 (heartbeat) | 18 B | 109 B |
| 5 | 53 B | 607 B |
| 10 | 88 B | 1,166 B |

18 B fixed + 7 B per hazard. Positions are rounded to 0.1 m because localization error is already
≥ 6 cm. Every packet is self-contained and protected by CRC-16, so one lost packet never corrupts another.

## Design principles (space flight software)

- **No cloud, no network dependency on board**: all inference runs locally on CPU.
- **One configuration file**: no magic numbers scattered in the code.
- **Bounded, predictable work per frame**: fixed input size, fixed-size packet records.
- **Fail safe**: corrupted or truncated packets are rejected (CRC), hazards that cannot be encoded are skipped, never sent wrong; rays near the horizon are rejected instead of producing huge ranges.
- **Every number is measured and tested**: `scripts/run_checks.py` regenerates all figures above.

## Training summary

Data: Artificial Lunar Rocky Landscape (rendered, rocks) + Roboflow Apollo "Lunar Rover" (real photos,
rocks and craters), 80/10/10 split. Model v1 = YOLO11n fine-tuned 60 epochs at 320 px. A rebalanced
fine-tune (real ×2, craters ×3) raised crater recall 35% → 40% but lowered overall accuracy; v1 was kept.
See `docs/results/experiments.csv`.

## Status

| Component | Status |
|---|---|
| Perception (detector, ONNX, benchmark) | Done |
| Localization (math, error model, tests) | Done |
| Telemetry packet format (encoder, decoder, CRC, tests) | Done |
| Hazard map (merge repeated sightings, IDs, priority) | In progress |
| Lossy link emulator (loss, bandwidth, blackouts) | In progress |
| Ground station + Mission Control dashboard | In progress |
| End-to-end run: video → dashboard | In progress |

## Team

<!-- names and roles -->
