# ATEC'26 — Specification Summary
**Challenge:** Project C.R.A.T.E.R. — *Constrained Reconnaissance, Autonomy & Telemetry in Extreme Regions*
**Dates:** October 10–11, 2026 · Kick-off 8:00 AM · Venue: ESPRIT
**Source:** `Atec'26 Specification Book.pdf` (11 pages)

---

## 1. Context (why)
Small autonomous rovers, like the ones in NASA's CADRE mission, map permanently shadowed lunar craters. A crater has none of the usual infrastructure: no GPS, no cellular network and no cloud. Lunar regolith absorbs high-frequency radio, so a rover that finds a hazard or an ice deposit during a blackout **can't stream heavy data back to base**.

A rover in this situation needs **resilient edge AI** that can:
1. Process raw visual sensor data **locally**
2. Turn what it sees into **spatial coordinates**
3. Compress telemetry into **minimal payloads**
4. Relay those payloads to Mission Control over **lossy networks**

## 2. The mission (what we must build)
> Propose and prototype a **software system** that lets autonomous lunar scouts detect environmental hazards, estimate their locations, send telemetry over constrained networks and show actionable data to Mission Control.

### The four core pillars
| Pillar | What it means |
|---|---|
| **Perceive** | Process visual sensor feeds locally and detect surface hazards in real time |
| **Localize** | Convert visual detections into grid coordinates relative to the **landing site** |
| **Transmit** | Compress and send telemetry over an **emulated low-bandwidth, lossy network** |
| **Visualize** | Build a Mission Control dashboard that shows incoming spatial data **and how fresh it is** |

### Fixed vs. flexible
| Fixed (non-negotiable) | Flexible (our choice) |
|---|---|
| The system is **100% software-emulated** (no hardware) | AI models |
| Core data flow: **Perception → Telemetry → Dashboard** | Serialization methods |
| The focus is on operating under **severe bandwidth constraints** | Network protocols |
| | Mathematical approaches |
| | Visualization style |
| | Programming languages |

## 3. Core functional objectives (hard requirements)
The team must implement the whole pipeline itself and be able to defend its architecture.

1. **Onboard perception:** detect features or hazards from visual feeds locally. **External cloud APIs are not allowed.**
2. **Spatial awareness:** project detections into spatial coordinates relative to a reference origin.
3. **Low-bandwidth communication:** serialize coordinate telemetry in **ultra-compact formats** suited to degraded radio links.
4. **Resilient data transport:** deliver telemetry through **team-implemented sockets or scripts that simulate signal dropouts and packet loss**.
5. **Mission Control interface:** show incoming spatial data on a clear dashboard that tells operators how old the data is (**data age**) and how uncertain it is (**uncertainty**).

## 4. Architecture toolbox (allowed and suggested)
Any stack is allowed, as long as **each choice is justified by the mission constraints**.
- **Perception:** object detection models, classical computer vision, or lightweight custom classifiers
- **Spatial math:** kinematic transforms, geometric projection, frame-transformation libraries (e.g. ROS `tf2`)
- **Serialization:** compact binary structs, custom byte arrays, lightweight protocols
- **Dashboard:** web frameworks (React, Vue, HTML/JS), visualization libraries, custom operator UIs

## 5. Evaluation process and deliverables

### Phase 1: Technical preselection
- At the **code freeze deadline**, each team submits its **code repository** and **automated test outputs**.
- The jury benchmarks every pipeline and grades it on 5 technical categories.
- The **top 7 teams** become finalists.

### Phase 2: Live pitch (finalists only)
- **7-minute** technical presentation followed by **3 minutes** of Q&A.
- The **end-to-end pipeline must run live**.
- The team must defend its architectural and **mathematical** choices.

### Judging rubric
| Stage | Category | Weight | What is evaluated |
|---|---|---|---|
| Preselection | Edge perception efficiency | 15% | **FPS**, tracking accuracy, and resource use on constrained compute |
| Preselection | Spatial & kinematic math | 15% | Accuracy and logical defense of the **2D pixel → 3D spatial** transform |
| Preselection | Telemetry & protocol efficiency | 15% | Creativity in byte compression/serialization; **payload survival through lossy links** |
| Preselection | Mission Control dashboard | 15% | Real-time ingestion, UI clarity, representation of **data aging and uncertainty** |
| Preselection | Systems integration & space compliance | 15% | **Automated** flow from video ingestion to the dashboard; adherence to **space flight-software principles** |
| Finals | Pitch & technical Q&A | 25% | Clarity, quality of the live demo, defense of choices |

The five preselection categories are weighted equally, so a weak pillar costs as much as a strong one earns. All five need to work.

---

## 6. What this means in practice (deliverables checklist)
- [ ] **Video/image input:** a simulated rover camera feed (recorded video or a synthetic lunar scene) fed in automatically
- [ ] **Local detector** (e.g. a small YOLO/ONNX model or classical CV) that detects craters, rocks and hazards and **reports its FPS and CPU/RAM use**
- [ ] **Tracking:** keep track IDs across frames, since tracking accuracy is graded
- [ ] **Pixel → 3D projection:** pinhole camera model with intrinsics/extrinsics, a ground-plane intersection, and rover pose → coordinates in the landing-site frame. Document the math and the error estimate.
- [ ] **Compact binary telemetry:** e.g. a fixed-size `struct` of a few bytes per detection (quantized coordinates, class, confidence, timestamp, sequence number) plus a CRC. Measure **bytes per message** against a JSON baseline.
- [ ] **Lossy link emulator** (our own code): configurable packet loss, latency, jitter, bandwidth cap and blackout windows. Add resilience: sequence numbers, ACK/retransmit or redundancy/FEC, and priority for critical hazards.
- [ ] **Mission Control dashboard:** live map of the landing-site grid, hazard markers whose **color or opacity shows data age**, **uncertainty ellipses**, link-status indicators (loss %, latency, last packet received)
- [ ] **End-to-end automation:** a single command that starts the whole pipeline
- [ ] **Flight-software principles:** bounded memory and no unbounded allocation in loops, watchdog/heartbeat, graceful degradation, deterministic timing, error handling, logging. NASA's "Power of 10" rules are a useful reference.
- [ ] **Automated tests + outputs** (this is required for preselection): perception benchmark, projection accuracy, serialization round-trip, delivery rate under X% loss
- [ ] **Repository + README** that explains the architecture and justifies each choice
- [ ] **Pitch** (if we reach the finals): 7 min with a live demo, and answers ready for questions on the math and protocol choices

## 7. Points the spec leaves unclear (to ask the organizers)
- The exact **code freeze** time
- Whether a **dataset or video** is provided, or whether we pick our own lunar imagery
- The **target hardware** for the FPS and resource benchmark
- The expected format of the **"automated test outputs"**
