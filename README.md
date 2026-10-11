# C.R.A.T.E.R — Constrained Reconnaissance, Autonomy & Telemetry in Extreme Regions

**ATEC'26** · A 100% software-emulated autonomous lunar scout: detects surface hazards locally on board, calculates their spatial coordinates relative to the landing site, fuses repeat sightings, and transmits ultra-compact telemetry to Mission Control over a severely degraded radio link.

```
┌─────────────────────────────────────── ONBOARD ROVER ────────────────────────────────────────┐
│ Camera Frames ──► PERCEIVE ──► LOCALIZE ──► HAZARD MAP ──► TRANSMIT                          │
│                   YOLO11n       Pinhole +     Fusion +       Binary Packets                   │
│                   ONNX, 1 CPU   Ground Plane  Priority       18B + 7B/hazard, CRC             │
└──────────────────────────────────────────────┬────────────────────────────────────────────────┘
                                               │ (UDP Socket Relay)
                                               ▼
                        ┌──────────────────────────────────────────────┐
                        │              LOSSY RADIO LINK                │
                        │    Burst Loss (Gilbert-Elliott), Latency     │
                        │    Jitter, Bandwidth Cap, Blackout Windows   │
                        └──────────────────────┬───────────────────────┘
                                               │
                                               ▼
┌───────────────────────────────────── MISSION CONTROL ─────────────────────────────────────────┐
│ GROUND STATION ──────────────────────────────► MISSION CONTROL DASHBOARD                     │
│ UDP Ingestion, CRC-16 Check,                   NASA Glassmorphism UI, 2D Grid, Interactive    │
│ Sequence Gap Analysis, Data Age Tracking       Zoom/Pan, Metric Ruler, Sparklines, PiP Video │
└───────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ Quick Start

```bash
# 1. Setup Python virtual environment
python -m venv .venv
source .venv/bin/activate            # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Run automated validation suite (produces all test outputs in test_outputs/)
python scripts/run_checks.py

# 3. Launch the live end-to-end mission simulation
python scripts/run_demo.py --source data/sample_images/video.mp4

# 4. Open Mission Control in your browser
# -> http://127.0.0.1:8000
```

> **Zero Cloud Dependency**: The entire rover pipeline runs strictly on CPU with `numpy`, `opencv-python`, and `onnxruntime` — no PyTorch, no GPU, no internet connection required.

---

## 📁 Repository Layout

| Path | Component | Description |
|---|---|---|
| `crater/perception.py` | **Perception** | Onboard hazard detector: grayscale + CLAHE preprocessing, YOLO11n ONNX inference on 1 CPU thread, NMS, `foot_px` ground contact extraction |
| `crater/localization.py` | **Localization** | 2D pixel → 3D landing-site coordinates ($E, N$), pinhole intrinsics, extrinsics, flat ground intersection ($z=0$), covariance error budget |
| `crater/hazard_map.py` | **Hazard Map** | Spatial tracking across frames: inverse-variance fusion ($1/\sigma^2$), persistent IDs, bounded memory (`max_tracks=256`), size & range filtering, severity priority |
| `crater/telemetry.py` | **Telemetry** | Ultra-compact binary serializer: 18B header/rover state + 7B per hazard, CRC-16/CCITT integrity protection |
| `crater/link.py` | **Radio Link** | Lossy network emulator: UDP relay (`LinkEmulator`) & fast virtual-time model (`LinkModel`) with Gilbert-Elliott burst loss, latency, jitter, bandwidth limits, blackouts |
| `crater/ground.py` | **Ground Station** | Mission Control ingestion: UDP receiver, CRC verification, sequence gap loss estimation, real-time data age calculation, HTTP API, MJPEG camera server |
| `web/dashboard.html` | **Mission Control** | High-fidelity NASA glassmorphism interface: interactive 2D grid map, zoom/pan, dynamic scale ruler bar, camera FOV frustum, HUD tooltips, telemetry sparklines, bottom-left PiP video |
| `scripts/run_demo.py` | **Demo Runner** | One-command launcher orchestrating the entire end-to-end mission pipeline |
| `scripts/run_checks.py` | **Checks & Benchmarks** | Generates all automated jury test outputs in `test_outputs/` |
| `scripts/live_localize.py` | **CV Visualizer** | Standalone real-time localizer with annotated video stream & top-view mini-map |
| `config/mission.json` | **Configuration** | Single source of truth for camera, mount, detector, rover, telemetry, and link parameters |
| `models/crater_detector.onnx`| **Deployed AI** | Fine-tuned lightweight YOLO11n model (320px) |
| `tests/` | **Unit Tests** | Complete pytest suite (28 automated tests covering all modules) |
| `test_outputs/` | **Jury Deliverables** | Generated benchmarks: `SUMMARY.md`, `junit.xml`, `delivery_report.json`, `localization_error_budget.csv`, `telemetry_report.json`, `perception_benchmark.json` |

---

## 📊 Measured Benchmark Results

### 1. Edge Perception (YOLO11n, 320 px, grayscale + CLAHE)
- **Model Size & Parameters**: 2.6 M parameters / 1.6 GFLOPs (10.5 MB ONNX).
- **Inference Speed (1 CPU thread)**: **32.8 ms/frame ≈ 30 FPS** on CPU.
- **Resource Footprint**: < 15 MB RAM overhead; at 0.1 m/s (2 inferences/sec), the rover requires only ~65 ms of CPU per second (**~6.5% of one core**).
- **Detection Accuracy**: mAP50 = 0.49 (full test set); Recall at conf 0.25: **88% rocks**, **93% craters**.

### 2. Spatial Localization & Error Budget (640×480, HFOV 90°, mount height 0.5 m, pitch 25°)
Linearized first-order error propagation ($\Sigma = J S J^T$) evaluated against 20,000-sample Monte Carlo verification:

| Distance from Camera | Along-Range $1\sigma$ | Cross-Range $1\sigma$ | 95% Confidence Semi-Major Axis |
|---|---|---|---|
| **2 m** | 0.11 m | 0.06 m | 0.29 m |
| **5 m** (Trusted Boundary) | **0.56 m** | **0.11 m** | **1.37 m** |
| **8 m** | 1.40 m | 0.16 m | 3.44 m |

*Camera pitch dominates the error budget; error grows quadratically with range ($D^2$). Rays flatter than 3° below the horizon are rejected.*

### 3. Telemetry Serialization Efficiency
Every packet is self-contained with quantized coordinates (0.1 m resolution) and protected by CRC-16:

| Hazards per Packet | Compact Binary | JSON Equivalent | Compression Factor |
|---|---|---|---|
| **0 (Heartbeat)** | **18 B** | 109 B | **6.1×** smaller |
| **5 hazards** | **53 B** | 607 B | **11.5×** smaller |
| **10 hazards** | **88 B** | 1,166 B | **13.3×** smaller |

### 4. Link Resilience & Hazard Delivery (Virtual-Time 5-Minute Mission)
Replayed through `LinkModel` (1.3 s latency, 9600 bit/s, burst factor 3.0, 15-second total blackout):

| Configured Packet Loss | Hazards Confirmed Onboard | Hazards Delivered at Base | Delivery Ratio | Median Latency | Mean Position Error |
|---|---|---|---|---|---|
| **0%** | 20 / 20 | 20 / 20 | **100%** | 12.8 s | 0.16 m |
| **10%** | 20 / 20 | 20 / 20 | **100%** | 12.9 s | 0.21 m |
| **30%** | 20 / 20 | 20 / 20 | **100%** | 12.8 s | 0.16 m |
| **60%** (Severe Degradation) | 20 / 20 | 20 / 20 | **100%** | 14.0 s | 0.16 m |

*Even under 60% burst loss and a 15-second blackout window, 100% of confirmed hazards reached Mission Control with sub-20 cm mean spatial error.*

---

## 🛡️ Hazard Discrimination & Filtering Policy

To prevent telemetry saturation on low-bandwidth links, the scout enforces strict **physical obstacle filtering**:

1. **Wheel Clearance Thresholds**:
   - The rover's wheel diameter is 0.2 m with 0.1 m ground clearance.
   - **Rocks**: Must be **BOTH** tall enough ($\ge 0.1\text{ m}$) **AND** wide enough ($\ge 0.2\text{ m}$) to endanger the chassis (`rock_rule: "all"`). Small pebbles and flat rocks that can be safely driven over are treated as non-hazardous regolith and **never transmitted**.
   - **Craters**: Must be wide enough ($\ge 0.3\text{ m}$) to trap wheels.
2. **Trusted Distance Boundary**:
   - Only detections within $5.0\text{ m}$ of the rover are eligible for telemetry; distant noisy background objects are suppressed.
3. **Temporal Multi-Sighting Confirmation**:
   - Hazards must be confirmed over multiple consecutive frames (`confirm_hits = 5`) before transmission, preventing false-positive noise.

---

## 🖥️ Mission Control Dashboard Features

The web interface (`web/dashboard.html`) delivers a modern, high-tech NASA/JPL flight-instrument experience:

- **Interactive 2D Grid Canvas**:
  - Full mouse-wheel zoom ($0.2\times$ to $8\times$) and click-and-drag pan.
  - **Dynamic Metric Scale Ruler**: Bottom-right bar automatically recalculates meters per pixel on zoom.
  - **Camera Vision Frustum (FOV 90°)**: Visualizes the rover's optical cone and 5 m trusted range boundary.
  - **One-Click Recenter**: Instant reframing on the rover.
- **Data Aging & Uncertainty Visualization**:
  - Hazard markers glow bright neon when fresh (< 5s) and fade smoothly to translucent ghost contours as age approaches 60s+.
  - 95% confidence error ellipses dynamically displayed around every hazard.
- **Interactive Hazard Inspector**:
  - Hovering or clicking any hazard on the map opens an aerospace HUD card displaying exact coordinates, metric dimensions, range, $\pm 95\%$ uncertainty, and data freshness.
- **Real-Time Link Health & Sparklines**:
  - Live SVG sparkline graph showing 60-second packet loss % history.
  - Radio status chip (`NOMINAL`, `DEGRADED`, `LOST`) with flashing blackout alerts.
- **Picture-in-Picture (PiP) Rover Camera**:
  - Positioned in the **bottom-left corner**, keeping the primary navigation grid unobstructed.
  - Expandable, minimizable, with auto-reconnect and crisp annotated detections.

---

## 🛰️ Space Flight-Software Principles

1. **No Cloud / No Network Dependency**: Zero external API calls; all perception and spatial transforms execute locally on CPU.
2. **Bounded Memory**: Bounded track capacity (`max_tracks=256`), ring buffers for telemetry history (`maxlen=800`), and fixed-size packet records prevent memory leaks.
3. **Fail-Safe & Graceful Degradation**:
   - Packets corrupted by bit flips or truncation are rejected by CRC-16.
   - Rays near the horizon (< 3° depression) are rejected to prevent range divergence.
   - During signal dropouts or blackouts, Mission Control displays extrapolated last-known states with clear staleness indicators.
4. **Single Source of Truth**: All operational parameters are governed by `config/mission.json`.

---

## 📋 Project Status

| Pillar / Component | Specification Goal | Status |
|---|---|---|
| **Pillar 1: Perceive** | Real-time onboard detection, 1 CPU thread, ONNX runtime, CLAHE | **✅ Complete** |
| **Pillar 2: Localize** | 2D $\rightarrow$ 3D ground projection, error budget, 95% uncertainty ellipses | **✅ Complete** |
| **Pillar 3: Transmit** | Compact binary protocol (18B + 7B/hazard), CRC-16 integrity | **✅ Complete** |
| **Pillar 4: Hazard Map** | Multi-sighting fusion ($1/\sigma^2$), persistent IDs, bounded memory, size filtering | **✅ Complete** |
| **Network Link Emulator** | UDP relay with burst loss (Gilbert-Elliott), latency, jitter, blackouts | **✅ Complete** |
| **Mission Control Dashboard** | Interactive grid map, scale ruler, data aging, sparklines, PiP video | **✅ Complete** |
| **End-to-End Automation** | One-command launcher from camera video to live web dashboard | **✅ Complete** |
| **Validation & Benchmarks** | Automated pytest suite (28 tests) & jury deliverables in `test_outputs/` | **✅ Complete** |

---

## 👥 Team

**NILESAT** · ATEC'26 Lunar Scout Challenge

| Name | Role |
|---|---|
| **Kacem Barhoumi** | Team Member |
| **Youcef Rbai** | Team Member |
