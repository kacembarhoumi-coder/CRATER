# C.R.A.T.E.R automated test outputs

Generated 2026-10-11 04:10:54 on Windows-11-10.0.26200-SP0, Python 3.14.0

## Unit tests

`28 passed in 2.84s` (details: pytest_report.txt, junit.xml)

## Localization error budget (1 sigma)

| Distance (m) | Along-range (m) | Cross-range (m) |
|---|---|---|
| 1 | 0.06 | 0.055 |
| 2 | 0.112 | 0.064 |
| 3 | 0.214 | 0.077 |
| 4 | 0.364 | 0.092 |
| 5 | 0.557 | 0.108 |
| 6 | 0.795 | 0.124 |
| 7 | 1.076 | 0.141 |
| 8 | 1.4 | 0.159 |
| 9 | 1.768 | 0.176 |

## Telemetry packet size

| Hazards | Binary (B) | JSON (B) |
|---|---|---|
| 0 | 18 | 109 |
| 1 | 25 | 207 |
| 5 | 53 | 607 |
| 10 | 88 | 1166 |

## Perception benchmark

Source: `data/sample_images`. 10 frames, 1 CPU thread(s): **69.8 ms/frame (14.3 FPS)** including preprocessing; model 10.54 MB. Annotated images in images/.

## Delivery under burst loss (baseline: repetition only, no ACK)

| Loss | Hazards delivered | Latency median / p95 (s) | Mean position error (m) |
|---|---|---|---|
| 0% | 20/20 | 12.8 / 83.4 | 0.16 |
| 10% | 20/20 | 12.9 / 83.3 | 0.21 |
| 30% | 20/20 | 12.8 / 83.3 | 0.16 |
| 60% | 20/20 | 14.0 / 86.3 | 0.16 |
