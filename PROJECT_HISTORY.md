# Project History and Dissertation Record

> Living technical record for the MSc Major Project. This file is a place to capture the rationale for the project, the implementation history, test results, failures, fixes, design decisions, limitations, and dissertation-ready interpretations so that the final report can be written from an auditable record.

## 1. Working project title

**A Software-Only Deep Learning Pipeline for Multi-Modal Object Detection and Robotic Perception Using LiDAR and Camera Fusion Across Multiple Datasets**

## 2. Core research idea

The project investigates a software-only robotic perception pipeline that combines camera detections and LiDAR measurements, with emphasis on controlled comparison rather than simply running multiple models.

The intended final system compares:

- camera-only perception,
- LiDAR-only perception,
- camera + LiDAR fusion,
- conventional closed-set camera detection,
- open-vocabulary / multimodal camera perception,
- multiple LiDAR detectors,
- multiple fusion strategies,
- cross-dataset generalization,
- robustness and runtime.

A central experimental principle is to **change one component at a time while holding the rest of the pipeline fixed**. In particular, YOLOv8x and Falcon Perception should be compared using the same KITTI frames, ground-truth matching rules, LiDAR projection, and downstream fusion rule wherever possible.

---

# 3. Current research questions

## RQ1 — Multimodal fusion

How much does camera–LiDAR fusion improve robotic perception compared with LiDAR-only or camera-only baselines?

## RQ2 — Conventional vs open-vocabulary camera perception

How does an open-vocabulary / multimodal detector such as Falcon Perception compare with a conventional detector such as YOLO when the LiDAR detector and fusion strategy are held fixed?

## RQ3 — Cross-dataset behavior

How well does the pipeline transfer from KITTI to a second autonomous-driving dataset such as nuScenes?

## Optional RQ4 — Fusion strategy

How do early association, late/candidate fusion, semantic point enrichment, and BEV-based fusion differ in accuracy, robustness, and runtime?

---

# 4. Planned model and framework stack

The project deliberately reuses strong open-source implementations where appropriate and adds clearly attributable project-specific adapters, controlled evaluation, fusion experiments, robustness tests, and cross-dataset analysis.

## Main repositories / frameworks

1. **MMDetection3D — `open-mmlab/mmdetection3d`**
   - Main 3D framework candidate.
   - Dataset support, evaluation utilities, PointPillars and BEVFusion-related projects.

2. **Azitt/Lidar-CameraFusion-3DobjectDetection**
   - Initial KITTI camera–LiDAR baseline reference.
   - YOLOv8 + PV-RCNN++.
   - Includes early and late fusion ideas.

3. **Falcon Perception — `tiiuae/Falcon-Perception`**
   - Main open-vocabulary / multimodal camera experiment.
   - MLX backend supports Apple Silicon.

4. **OpenPCDet — `open-mmlab/OpenPCDet`**
   - LiDAR-only detection framework.
   - Candidate models: PV-RCNN++ and PointPillars.

5. **BEVFusion — `ADLab-AutoDrive/BEVFusion`**
   - Advanced fusion reference.
   - Expected to require older Linux/CUDA environment.

6. **MultiCorrupt — `ika-rwth-aachen/MultiCorrupt`**
   - Planned robustness experiments.

7. **NVIDIA Lidar AI Solution**
   - Stretch goal for optimized inference if suitable NVIDIA hardware is available.

## Additional technical references

- CLOCs
- PointPainting
- TransFusion
- YOLO–LiDAR fusion implementations

## Academic transparency rule

Any reused code or architecture must remain attributable to the original source and compatible with its license. The dissertation should clearly distinguish:

- reproduced external baseline,
- adapted external code,
- project-specific integration,
- project-specific evaluation,
- project-specific experimental contributions.

---

# 5. Hardware and environment decisions

## Development machine

- Apple Silicon MacBook Pro, M1 Pro, arm64.
- Primary local role: Python development, KITTI processing, calibration, projection, YOLO inference, Falcon MLX inference, adapter development, evaluation, plotting, and fusion logic.

## Important environment decision

CUDA-dependent frameworks must **not** be forced onto macOS. Docker on Apple Silicon does not provide an NVIDIA CUDA GPU.

The project therefore keeps separate environments and communicates between components through ordinary files / arrays / JSON where necessary.

### `azitt310`

Used for:

- Python 3.10,
- Ultralytics / YOLOv8x,
- KITTI camera–LiDAR baseline,
- MPS inference,
- fusion and evaluation scripts.

### `falcon310`

Used for:

- Python 3.10,
- Falcon Perception,
- MLX inference on Apple Silicon.

### CUDA-heavy work later

Expected to require Linux + NVIDIA GPU for:

- OpenPCDet,
- PV-RCNN++,
- some PointPillars configurations,
- BEVFusion,
- other `spconv`-dependent pipelines.

---

# 6. Repository organization

Main project directory:

```text
~/Major_Project/
├── multimodal-perception/       # this repository
├── external/                    # cloned external repositories
├── datasets/                    # datasets, not committed
├── model_outputs/               # generated model outputs, normally not committed
└── envs/                        # Python environments, not committed
```

Own repository structure includes:

```text
adapters/
calibration/
configs/
evaluation/
experiments/
fusion/
requirements/
results/
scripts/
tests/
visualization/
README.md
PROJECT_HISTORY.md
```

Large datasets, environments, model weights, and ordinary generated inference outputs should remain outside Git history. Small dissertation result CSVs and figures may be committed when they represent frozen experimental milestones.

---

# 7. KITTI dataset setup

KITTI was selected as the primary development dataset.

Local training data verified:

- `training/calib`: 7,481 files
- `training/label_2`: 7,481 files
- `training/image_2`: 7,481 files
- `training/velodyne`: 7,481 files

Testing data verified:

- `testing/calib`: 7,518 files
- `testing/image_2`: 7,518 files
- `testing/velodyne`: 7,518 files

The original ZIP archives were removed after successful extraction to recover disk space.

Frame `000149` became the main early debugging / visualization frame.

---

# 8. KITTI LiDAR-to-camera projection

A reusable KITTI projection pipeline was implemented.

Relevant calibration matrices:

- `P2`
- `R0_rect`
- `Tr_velo_to_cam`

The implementation:

1. reads Velodyne points,
2. converts points to homogeneous coordinates,
3. transforms Velodyne coordinates into rectified camera coordinates,
4. keeps points in front of the camera,
5. projects them with `P2`,
6. removes points outside the image,
7. retains forward camera depth `z` for each projected point.

Visual alignment on KITTI imagery was checked before using the projection in fusion experiments.

Important early commit:

- `3fef0bb` — **Add KITTI LiDAR to camera projection**

---

# 9. YOLOv8x + LiDAR fusion baseline

The initial project-owned fusion baseline used YOLOv8x detections and projected LiDAR points.

## Frozen V1 association method

For each relevant 2D detection:

1. project LiDAR points into the camera image,
2. retain projected points whose image coordinates fall inside the 2D detection box,
3. require at least three LiDAR returns,
4. estimate object forward depth as the **median** of the LiDAR camera-coordinate depths inside the box.

This method is intentionally simple and reproducible.

The median was chosen rather than the mean because it is less sensitive to isolated background points.

Initial visualization script:

- `scripts/fuse_yolov8_lidar.py`

Important commit:

- `95d3221` — **Add YOLOv8x LiDAR fusion baseline and depth evaluation**

---

# 10. Depth estimator study: V1 vs V2 vs V3

This became an important methodological sub-study. The aim was to determine whether a more complicated depth rule actually improved the original median-ROI estimator.

## V1 — Median ROI

Depth estimate:

```text
median(depth of projected LiDAR points inside the 2D detection box)
```

## V2 — Nearest substantial depth cluster

Projected LiDAR depths were sorted and split when a depth gap exceeded approximately 1.5 m. The method then selected a substantial near cluster.

Hypothesis:

- a near cluster might remove background contamination when the detector box spans several depth layers.

Result:

- the hypothesis did not generalize well;
- V2 often selected the wrong surface and created large underestimation errors.

## V3 — Edge-aware hybrid

Development-set error analysis showed severe failures for some heavily truncated edge detections.

V3 rule:

- if a detection box touched approximately the outer 2% of the image width, use V2;
- otherwise use V1.

This rule used only observable detection/image geometry and did not use ground-truth depth at inference time.

The method improved development metrics slightly but failed to improve the independent holdout set overall.

## Final decision

**V1 Median ROI remains the frozen depth baseline.**

This is a useful negative result: increasing algorithmic complexity did not automatically improve generalization.

---

# 11. Single-frame depth comparison

On KITTI frame `000149`, 9 YOLO detections were matched to KITTI `Car` ground truth.

Results:

| Method | MAE |
|---|---:|
| V1 Median ROI | **1.825 m** |
| V2 Nearest Cluster | 7.141 m |

V1 was substantially better on this frame.

---

# 12. Multi-frame depth evaluation

The experiment was then expanded beyond a single example.

Ground-truth matching used a 2D IoU threshold of **0.30** for the exploratory depth study.

## 10-frame development result

45 matched detections.

| Method | MAE | Median AE | RMSE |
|---|---:|---:|---:|
| V1 | **1.492 m** | **0.923 m** | **2.402 m** |
| V2 | 2.930 m | 1.409 m | 4.930 m |

## 50-frame development result

180 matched detections.

| Method | MAE | Median AE | RMSE |
|---|---:|---:|---:|
| V1 | **1.577 m** | **1.098 m** | **2.545 m** |
| V2 | 3.423 m | 1.451 m | 6.368 m |

## 100-frame development result

360 matched detections.

| Method | N | MAE | Median AE | RMSE |
|---|---:|---:|---:|---:|
| V1 | 360 | **1.898 m** | **1.265 m** | 3.750 m |
| V2 | 359 | 3.986 m | 1.684 m | 6.675 m |
| V3 | 360 | **1.862 m** | **1.265 m** | **3.656 m** |

V3 appeared slightly better than V1 on development data, so an independent holdout test was required before adopting it.

---

# 13. Independent depth holdout validation

A deterministic holdout was generated with random seed `42`, excluding the evenly spaced development indices.

100 separate frames were evaluated.

351 matched detections were available for V1/V3.

| Method | N | MAE | Median AE | RMSE |
|---|---:|---:|---:|---:|
| V1 | 351 | **1.954 m** | 1.318 m | **3.925 m** |
| V2 | 350 | 4.195 m | 1.631 m | 6.957 m |
| V3 | 351 | 1.962 m | **1.285 m** | 3.948 m |

## Decision

V3 did not improve holdout MAE or RMSE compared with V1.

Therefore:

**V1 was retained as the final simple depth-association baseline.**

This development-vs-holdout sequence should be emphasized in the dissertation as evidence that design choices were validated rather than selected solely from development performance.

Important commit:

- `28e01e3` — **Add edge-aware fusion ablation and holdout validation**

---

# 14. Depth error analysis

For the 100-frame development experiment:

## Mean signed error

Estimated depth minus KITTI ground-truth object `z`:

- V1 mean signed error: **-1.075 m**
- V2 mean signed error: **-3.674 m**

The negative bias is physically plausible because projected LiDAR points commonly lie on the visible front surface of a vehicle, whereas KITTI object location `z` represents the annotated 3D object location / center in camera coordinates.

Therefore the LiDAR ROI estimate and KITTI center depth are related but not exactly the same physical quantity.

## V1 distance-bin analysis

| GT distance | N | MAE | Median AE | RMSE |
|---|---:|---:|---:|---:|
| 0–20 m | 143 | 1.474 | 0.780 | 2.674 |
| 20–40 m | 144 | 1.878 | 1.407 | 2.999 |
| 40–60 m | 59 | 2.176 | 1.713 | 3.446 |
| 60+ m | 14 | 5.258 | 2.013 | 12.087 |

Performance deteriorates at long range, where projected LiDAR support becomes increasingly sparse and individual association errors become more damaging.

## V2 distance-bin analysis

| GT distance | N | MAE | Median AE | RMSE |
|---|---:|---:|---:|---:|
| 0–20 m | 143 | 1.888 | 1.060 | 3.059 |
| 20–40 m | 144 | 5.218 | 1.861 | 7.231 |
| 40–60 m | 59 | 5.345 | 1.896 | 8.534 |
| 60+ m | 14 | 7.234 | 2.020 | 14.757 |

V2 was particularly unstable at medium and long range.

---

# 15. Important V1 failure cases

Large errors from the 100-frame development analysis included:

| Frame | GT z | V1 depth | Absolute error | 2D match IoU | LiDAR points |
|---|---:|---:|---:|---:|---:|
| 007404 | 62.91 | 18.97 | 43.94 | 0.734 | 40 |
| 007102 | 49.76 | 30.73 | 19.03 | 0.338 | 58 |
| 006497 | 35.14 | 19.77 | 15.37 | 0.720 | 95 |
| 003173 | 3.31 | 17.43 | 14.12 | 0.713 | 713 |
| 002568 | 2.76 | 15.09 | 12.33 | 0.829 | 1,216 |
| 000755 | 29.08 | 17.29 | 11.79 | 0.887 | — |
| 003097 | 26.87 | 15.39 | 11.48 | — | — |
| 000831 | 1.65 | 11.21 | 9.56 | — | — |
| 004004 | 29.20 | 20.06 | 9.14 | — | — |
| 001435 | 40.58 | 31.49 | 9.09 | — | — |

Visual inspection identified several recurring mechanisms:

- multiple physical depth layers inside a rectangular detection ROI,
- overlapping/crowded vehicles,
- severe image-edge truncation,
- background LiDAR contamination,
- sparse distant LiDAR returns.

A key lesson is that a high number of LiDAR points does **not** guarantee a good depth estimate if those points belong to several surfaces.

---

# 16. Visibility and occlusion analysis

## Truncation bins — V1

| Truncation | N | MAE | Median AE | RMSE |
|---|---:|---:|---:|---:|
| 0–0.15 | 306 | 1.834 | 1.302 | 3.707 |
| >0.15–0.30 | 6 | 0.181 | 0.197 | 0.208 |
| >0.30–0.50 | 9 | 0.632 | 0.527 | 0.750 |
| >0.50 | 39 | 2.959 | 1.184 | 4.680 |

The middle truncation bins contain too few samples for strong general conclusions.

## Occlusion bins — V1

| Occlusion | N | MAE | Median AE | RMSE |
|---|---:|---:|---:|---:|
| 0 — fully visible | 194 | 1.603 | 1.347 | 2.455 |
| 1 — partly occluded | 97 | 1.676 | 0.904 | 4.807 |
| 2 — largely occluded | 59 | 3.504 | 1.653 | 5.290 |
| 3 — unknown | 10 | 0.296 | 0.217 | 0.367 |

Occlusion level 2 was associated with materially larger V1 errors.

---

# 17. Frozen depth result artifact

A dissertation-ready comparison artifact was generated:

- `results/kitti/depth/method_comparison/depth_method_comparison.csv`
- `results/kitti/depth/method_comparison/figures/depth_method_comparison.png`

Exact frozen table:

| Split | Method | N | MAE | Median AE | RMSE |
|---|---|---:|---:|---:|---:|
| Development | V1 Median ROI | 360 | 1.8978453 | 1.2646964 | 3.75031997 |
| Development | V2 Nearest Cluster | 359 | 3.9856579 | 1.6841375 | 6.67490273 |
| Development | V3 Edge-Aware Hybrid | 360 | 1.8618330 | 1.2646964 | 3.65573772 |
| Holdout | V1 Median ROI | 351 | 1.9538831 | 1.3175633 | 3.92486870 |
| Holdout | V2 Nearest Cluster | 350 | 4.1945221 | 1.6313400 | 6.95719580 |
| Holdout | V3 Edge-Aware Hybrid | 351 | 1.9618197 | 1.2846350 | 3.94787560 |

Important commit:

- `bc8b7cf` — **Add depth estimation comparison results**

---

# 18. Important methodological caveat for the depth study

The reported depth errors are **conditional on camera detections that were matched to KITTI Car annotations** at the chosen 2D IoU threshold.

They are not a complete object-detection metric.

The dissertation must not present these MAE/RMSE values as overall detection accuracy.

---

# 19. Falcon Perception integration

Falcon Perception was selected as the main open-vocabulary / multimodal camera component.

## Why Falcon was useful for this project

- It supports natural-language queries such as `car`.
- It represents a different perception paradigm from a conventional fixed-taxonomy YOLO detector.
- Its official repository includes an MLX path for Apple Silicon.
- It therefore allowed meaningful local experimentation without CUDA.

## Falcon environment

Environment:

```text
~/Major_Project/envs/falcon310
```

Verified:

- Python 3.10.21 arm64
- Falcon Perception import
- MLX import
- editable Falcon installation

---

# 20. First Falcon KITTI run

Official MLX detection was run on KITTI frame `000149` with query:

```text
car
```

The first run:

- fetched 17 model files,
- downloaded roughly 2.5 GB,
- loaded the MLX perception model in float16,
- produced 11 bounding boxes,
- produced 0 masks for the detection task,
- generated the annotated image `000149_car.jpg`.

Measured first-run timing:

- model loading: **124.94 s**
- preprocessing: approximately **0.015 s**
- generation: **4.98 s**
- total: **129.98 s**

The long first-run model load must not be interpreted as steady-state per-frame latency because it included initial model acquisition / initialization work.

Later cached model loading was approximately **0.6 s**.

---

# 21. Falcon adapter

Project-owned adapter:

- `adapters/falcon_adapter.py`

Purpose:

1. load Falcon with MLX,
2. execute a natural-language detection query,
3. parse Falcon bounding-box entries,
4. convert normalized center-format boxes to pixel `x1, y1, x2, y2`,
5. export a project-friendly JSON representation.

Important technical detail:

Falcon's official output path used here does **not** expose a conventional confidence score per bounding box.

Therefore the project must not invent scores or report confidence-ranked Falcon mAP without a supported scoring mechanism.

---

# 22. Falcon single-frame diagnostic

On KITTI frame `000149`:

- Falcon detections: 11
- KITTI `Car` ground-truth boxes: 10

Greedy one-to-one matching produced:

| 2D IoU threshold | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| 0.30 | 10 | 1 | 0 | 0.909 | 1.000 |
| 0.50 | 7 | 4 | 3 | 0.636 | 0.700 |
| 0.70 | 4 | 7 | 6 | 0.364 | 0.400 |

This was treated only as a diagnostic, not final accuracy.

---

# 23. Falcon 10-frame raw pilot

Falcon was then evaluated on 10 evenly spaced KITTI training frames while loading the model once.

Frames:

```text
000000
000831
001662
002493
003324
004155
004986
005817
006648
007480
```

Total KITTI Car ground truth across the split: **48**.

Raw Falcon produced **168** detections.

## Raw aggregate results

| IoU | TP | FP | FN | Precision | Recall | F1 | Mean matched IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.30 | 41 | 127 | 7 | 0.244 | 0.854 | 0.380 | 0.678 |
| 0.50 | 31 | 137 | 17 | 0.185 | 0.646 | 0.287 | 0.751 |
| 0.70 | 21 | 147 | 27 | 0.125 | 0.438 | 0.194 | 0.821 |

Mean Falcon generation time:

- **6.416 s/frame**
- approximate generation rate: **0.156 FPS**

This number refers to the measured Falcon generation section, not a fully standardized end-to-end benchmark.

---

# 24. Major Falcon failure discovered: duplicate / redundant boxes

Frame `006648` was an extreme failure case:

```text
KITTI GT cars = 9
Raw Falcon boxes = 101
Generation time = 25.97 s
```

The annotated image was inspected manually.

The result showed:

- many highly overlapping bounding boxes around the same vehicles,
- multiple redundant boxes in the right half of the image,
- several small questionable detections in background regions.

Therefore the poor raw precision was not simply because Falcon failed to find cars. A substantial part of the problem was **redundant score-free output**.

This was a critical diagnostic finding because it changed the correct next step from “discard Falcon” to “evaluate transparent score-free post-processing.”

---

# 25. Score-free Falcon deduplication

Because no conventional per-box confidence score was available, confidence-ranked NMS was not appropriate.

A project-owned score-free method was implemented:

1. calculate IoU between Falcon boxes,
2. build connected components where box pairs overlap above a selected threshold,
3. treat each connected component as a duplicate cluster,
4. replace the cluster with the coordinate-wise median bounding box.

This keeps the procedure deterministic and does not invent model confidence.

Three development thresholds were tested:

- 0.50
- 0.70
- 0.80

---

# 26. Falcon deduplication development ablation

## Raw Falcon

| Evaluation IoU | Precision | Recall | F1 | FP |
|---|---:|---:|---:|---:|
| 0.30 | 0.244 | 0.854 | 0.380 | 127 |
| 0.50 | 0.185 | 0.646 | 0.287 | 137 |
| 0.70 | 0.125 | 0.438 | 0.194 | 147 |

## Dedup threshold 0.50

| Evaluation IoU | Precision | Recall | F1 | FP |
|---|---:|---:|---:|---:|
| 0.30 | **0.526** | 0.854 | **0.651** | **37** |
| 0.50 | **0.397** | 0.646 | **0.492** | **47** |
| 0.70 | **0.269** | 0.438 | **0.333** | **57** |

## Dedup threshold 0.70

| Evaluation IoU | Precision | Recall | F1 |
|---|---:|---:|---:|
| 0.30 | 0.506 | 0.854 | 0.636 |
| 0.50 | 0.383 | 0.646 | 0.481 |
| 0.70 | 0.259 | 0.438 | 0.326 |

## Dedup threshold 0.80

| Evaluation IoU | Precision | Recall | F1 |
|---|---:|---:|---:|
| 0.30 | 0.482 | 0.854 | 0.617 |
| 0.50 | 0.365 | 0.646 | 0.466 |
| 0.70 | 0.247 | 0.438 | 0.316 |

## Development decision

Threshold **0.50** was the best of the tested score-free deduplication thresholds.

At evaluation IoU 0.30:

```text
raw FP: 127
0.50 dedup FP: 37
raw F1: 0.380
0.50 dedup F1: 0.651
```

That is approximately a 71% relative increase in F1 on this development pilot.

For extreme frame `006648`:

```text
raw boxes = 101
0.50 deduplicated boxes = 17
GT cars = 9
```

---

# 27. Independent Falcon dedup holdout

The dedup threshold was **not** accepted solely from development data.

An independent 10-frame holdout was generated by excluding the development indices and sampling from the remaining KITTI training frames with deterministic random seed `42`.

Holdout frames:

```text
000642
000666
000704
001506
003236
003280
004891
005215
005783
006419
```

Total KITTI Car ground truth: **45**.

## Raw Falcon holdout

| IoU | TP | FP | FN | Precision | Recall | F1 | Mean matched IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.30 | 42 | 33 | 3 | 0.560 | 0.933 | 0.700 | 0.754 |
| 0.50 | 38 | 37 | 7 | 0.507 | 0.844 | 0.633 | 0.790 |
| 0.70 | 30 | 45 | 15 | 0.400 | 0.667 | 0.500 | 0.836 |

## Falcon + dedup 0.50 holdout

| IoU | TP | FP | FN | Precision | Recall | F1 | Mean matched IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.30 | 42 | 30 | 3 | **0.583** | 0.933 | **0.718** | 0.751 |
| 0.50 | 38 | 34 | 7 | **0.528** | 0.844 | **0.650** | 0.787 |
| 0.70 | 29 | 43 | 16 | 0.403 | 0.644 | 0.496 | 0.840 |

## Decision

The 0.50 dedup rule generalized in a qualified way:

- precision and F1 improved at IoU 0.30,
- precision and F1 improved at IoU 0.50,
- recall was preserved at those thresholds,
- strict IoU 0.70 F1 fell slightly from 0.500 to 0.496.

Therefore:

**Falcon raw output remains a baseline, while Falcon + score-free dedup 0.50 is the standardized downstream version.**

The strict-IoU tradeoff must be reported rather than hidden.

Important commit:

- `48e50ed` — **Add Falcon Perception adapter and KITTI evaluation**

---

# 28. Controlled YOLOv8x camera evaluation

YOLOv8x was evaluated on **exactly the same 10 development and 10 holdout frames** used for Falcon.

YOLO settings:

```text
weights = yolov8x.pt
confidence threshold = 0.25
device = MPS
class = car
```

The same greedy one-to-one matching implementation and the same evaluation IoU thresholds were used:

- 0.30
- 0.50
- 0.70

This was designed as a controlled fixed-operating-point pilot comparison.

Important commit:

- `ddd1aa2` — **Add controlled YOLOv8x KITTI camera evaluation**

---

# 29. YOLOv8x development pilot results

| IoU | TP | FP | FN | Precision | Recall | F1 | Mean matched IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.30 | 45 | 29 | 3 | 0.608 | 0.938 | 0.738 | 0.710 |
| 0.50 | 40 | 34 | 8 | 0.541 | 0.833 | 0.656 | 0.746 |
| 0.70 | 24 | 50 | 24 | 0.324 | 0.500 | 0.393 | 0.838 |

Measured mean prediction time including the first warm-up-heavy frame:

- **0.197 s/frame**
- approximately **5.079 FPS**

The first frame took 1.433 s while subsequent frames were approximately 0.06 s, so the raw development mean is not a representative steady-state runtime value.

---

# 30. YOLOv8x independent holdout results

| IoU | TP | FP | FN | Precision | Recall | F1 | Mean matched IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.30 | 41 | 21 | 4 | **0.661** | 0.911 | **0.766** | 0.767 |
| 0.50 | 36 | 26 | 9 | **0.581** | 0.800 | **0.673** | 0.815 |
| 0.70 | 30 | 32 | 15 | **0.484** | 0.667 | **0.561** | 0.856 |

Mean prediction time on holdout:

- **0.058 s/frame**
- approximately **17.216 FPS**

This is much more representative of warmed-up YOLO inference on this machine than the development average.

---

# 31. Controlled YOLOv8x vs Falcon camera finding

For the fair pilot comparison, Falcon means **Falcon + score-free dedup 0.50**.

## Holdout comparison

| Metric | YOLOv8x | Falcon + Dedup | Better value |
|---|---:|---:|---|
| Precision @ IoU 0.30 | **0.661** | 0.583 | YOLOv8x |
| Recall @ IoU 0.30 | 0.911 | **0.933** | Falcon |
| F1 @ IoU 0.30 | **0.766** | 0.718 | YOLOv8x |
| Precision @ IoU 0.50 | **0.581** | 0.528 | YOLOv8x |
| Recall @ IoU 0.50 | 0.800 | **0.844** | Falcon |
| F1 @ IoU 0.50 | **0.673** | 0.650 | YOLOv8x |
| F1 @ IoU 0.70 | **0.561** | 0.496 | YOLOv8x |

## Interpretation

The current pilot suggests:

- YOLOv8x is the stronger conventional KITTI car detector at this fixed operating point,
- YOLOv8x has higher precision and overall F1,
- Falcon has slightly higher recall at IoU 0.30 and 0.50 on the independent holdout,
- Falcon offers open-vocabulary flexibility that YOLOv8x does not provide in the same way,
- Falcon's raw output required explicit duplicate handling,
- Falcon is substantially slower locally than YOLOv8x.

The dissertation should **not** frame the goal as making Falcon win. The academically useful result is the tradeoff between conventional detector precision/efficiency and open-vocabulary flexibility/recall.

---

# 32. Camera comparison result artifact

Frozen pilot artifacts:

- `results/kitti/camera/detector_comparison/camera_detector_comparison.csv`
- `results/kitti/camera/detector_comparison/figures/camera_detector_comparison.png`

Figure title:

**KITTI Camera Detection Pilot: YOLOv8x vs Falcon Perception**

Important commit:

- `f63ea07` — **Add YOLOv8x and Falcon camera comparison results**

---

# 33. Critical camera-evaluation caveat

The camera comparison is currently a **controlled fixed-threshold pilot** on:

- 10 development frames,
- 10 independent holdout frames.

It is **not official KITTI AP/mAP**.

The current matcher:

- evaluates `Car`,
- uses greedy one-to-one assignment,
- uses chosen 2D IoU thresholds,
- does not yet reproduce the full official KITTI evaluation protocol including official difficulty categories and `DontCare` handling.

Recommended dissertation wording:

> These results represent a controlled fixed-operating-point pilot evaluation on 10 development and 10 independent holdout KITTI frames; they are not official KITTI AP/mAP results.

A larger and/or standardized evaluation should be added later where feasible.

---

# 34. Runtime caveat

Do not report the current raw timing numbers as a perfectly standardized speed comparison.

Current values include:

- YOLO prediction timing,
- Falcon generation timing,
- different model execution structures,
- warm-up effects.

A later formal benchmark should:

1. load each model once,
2. perform one or more warm-up frames,
3. time the same number of frames,
4. clearly define preprocessing, inference/generation, post-processing, and end-to-end latency,
5. report hardware and precision.

Therefore statements such as “YOLO is exactly X times faster” should be avoided until standardized timing is complete.

---

# 35. Problems encountered and how they were resolved

| Problem | Diagnosis | Action / fix | Final decision |
|---|---|---|---|
| Azitt requirements include CUDA-specific `spconv-cu114` | Full original stack is not compatible with Apple Silicon as-is | Use Mac for camera, projection, evaluation and software fusion; reserve CUDA-heavy LiDAR detector for Linux/NVIDIA | Do not force CUDA stack onto Mac |
| Azitt code contains hard-coded `.cuda()` paths | GPU backend assumption in reference implementation | Reproduce camera/fusion logic independently where possible | Full unchanged PV-RCNN++ baseline remains future GPU work |
| Simple 2D ROI can contain background LiDAR | Rectangular detector boxes span multiple surfaces | Tested V2 nearest-cluster estimator | V2 was worse; retain median ROI |
| V3 edge-aware heuristic looked better on development data | Possible overfitting to observed failure cases | Created independent 100-frame holdout | V3 did not improve holdout MAE/RMSE; retain V1 |
| Very large V1 depth errors | Crowding, truncation, mixed surfaces, sparse distant returns | Per-frame, distance, truncation and occlusion analyses | Record these as known baseline limitations |
| Falcon first run very slow | Model download / initialization | Re-run after model cache | Separate model load from per-frame generation timing |
| Falcon produced 101 boxes for 9 GT cars on frame 006648 | Visual inspection showed many duplicate/redundant boxes | Implemented score-free IoU clustering + median representative | Dedup 0.50 selected on development and checked on holdout |
| Falcon has no conventional box confidence in the used output path | Cannot honestly apply confidence-ranked NMS or mAP ranking | Use score-free dedup and fixed-threshold metrics | Never invent Falcon confidence values |
| Dedup 0.50 strong on development but could overfit | Development threshold selection bias | Independent 10-frame holdout | Keep dedup 0.50 with explicit strict-IoU tradeoff |
| YOLO development mean runtime looked slow | First prediction was MPS warm-up outlier | Inspect per-frame timing and holdout steady state | Formal warm-up benchmark required later |
| Camera pilot is only 20 frames total | Small sample and custom matcher | Treat result as pilot; scale/standardize later | Do not call it official KITTI performance |

---

# 36. Key design decisions now frozen

Unless a later experiment provides strong evidence to change them:

1. **KITTI remains the primary development dataset.**
2. **nuScenes remains the planned secondary / cross-dataset dataset.**
3. **YOLOv8x remains the reproduced conventional camera baseline.**
4. **Falcon Perception remains the main open-vocabulary camera experiment.**
5. **Falcon downstream boxes use score-free IoU deduplication at 0.50**, while raw Falcon is retained as an ablation baseline.
6. **V1 Median ROI remains the frozen simple projected-LiDAR depth association method.**
7. The controlled YOLO-vs-Falcon multimodal comparison holds LiDAR projection and association fixed.
8. PointPillars is planned as an additional LiDAR baseline.
9. PV-RCNN++ remains relevant because of the Azitt reference pipeline.
10. BEVFusion remains the advanced fusion comparison target.
11. Robustness should include sensor degradation / dropout, range, occlusion, lighting and/or calibration perturbation where feasible.

---

# 37. Controlled multimodal comparison design

The controlled comparison isolates the camera detector variable while keeping the downstream LiDAR procedure fixed:

```text
YOLOv8x box
   ↓
same KITTI LiDAR projection
   ↓
same minimum-point rule
   ↓
same median-ROI depth association

versus

Falcon + dedup box
   ↓
same KITTI LiDAR projection
   ↓
same minimum-point rule
   ↓
same median-ROI depth association
```

This experiment has now been completed on matched development and independent holdout splits. The final results are recorded in Section 47 and in `results/VALIDATED_RESULTS.md`.

---

# 38. Planned advanced experiments

## Camera

- YOLOv8x reproduced baseline
- Falcon Perception
- one verified current conventional YOLO-family model later, if useful

## LiDAR

- PV-RCNN++
- PointPillars

## Fusion

- projected ROI early association
- candidate / CLOCs-style fusion
- PointPainting-style semantic point enrichment
- BEVFusion

## Datasets

- KITTI primary
- nuScenes secondary, beginning with mini / selected subset
- Glasgow qualitative RGB-only material only if useful and clearly marked as lacking paired LiDAR/calibration
- Waymo only as a stretch goal

## Robustness

Potential axes:

- object distance,
- occlusion,
- truncation,
- image degradation / low lighting,
- LiDAR sparsification,
- simulated sensor loss,
- calibration perturbation / misalignment,
- cross-dataset generalization.

## Runtime

- warmed-up per-frame latency,
- end-to-end latency,
- FPS,
- model-loading cost separately.

---

# 39. Current master status

## Completed / substantially completed

- [x] Project topic and architecture defined
- [x] Main repository stack selected
- [x] KITTI downloaded and verified
- [x] KITTI LiDAR-to-camera projection
- [x] YOLOv8x camera inference
- [x] YOLOv8x + projected LiDAR median-depth baseline
- [x] V1/V2/V3 depth estimator study
- [x] 100-frame development depth analysis
- [x] 100-frame independent depth holdout
- [x] Distance / visibility / failure analysis
- [x] Frozen depth comparison CSV + figure
- [x] Falcon MLX setup
- [x] Falcon independent KITTI inference
- [x] Falcon adapter
- [x] Falcon single-frame diagnostic
- [x] Falcon 10-frame raw evaluation
- [x] Falcon duplicate failure diagnosis
- [x] Falcon dedup threshold ablation
- [x] Falcon independent dedup holdout
- [x] Controlled YOLOv8x camera evaluation on identical splits
- [x] Frozen YOLOv8x vs Falcon camera CSV + figure

## Current / next

- [x] Falcon + LiDAR fusion using frozen median-ROI method
- [x] Multi-frame Falcon + LiDAR development evaluation
- [x] Independent Falcon + LiDAR holdout evaluation
- [x] Controlled YOLOv8x + LiDAR development evaluation
- [x] Independent YOLOv8x + LiDAR holdout evaluation
- [x] Controlled YOLOv8x + LiDAR vs Falcon + LiDAR comparison
- [x] Frozen comparison CSV + figure

Next technical priorities:

- [ ] current released YOLO-family comparison, if justified
- [ ] PointPillars LiDAR-only 3D baseline
- [ ] one stronger multimodal fusion method
- [ ] nuScenes cross-dataset evaluation
- [ ] robustness experiments
- [ ] standardized runtime benchmark

## Still planned

- [ ] larger / standardized camera evaluation
- [ ] PointPillars
- [ ] PV-RCNN++ / OpenPCDet GPU stage
- [ ] CLOCs-style fusion
- [ ] PointPainting / semantic fusion
- [ ] MMDetection3D integration
- [ ] BEVFusion
- [ ] nuScenes
- [ ] cross-dataset evaluation
- [ ] robustness experiments
- [ ] standardized runtime benchmark
- [ ] final ablations
- [ ] final demo
- [ ] reproducibility cleanup
- [ ] final dissertation tables, results and discussion

---

# 40. Important commits to date

| Commit | Description |
|---|---|
| `8782ae6` | Initial project setup |
| `3e997a6` | Add initial multimodal perception project structure |
| `3fef0bb` | Add KITTI LiDAR to camera projection |
| `95d3221` | Add YOLOv8x LiDAR fusion baseline and depth evaluation |
| `dc017df` | Add multi-frame KITTI depth and visibility analysis |
| `28e01e3` | Add edge-aware fusion ablation and holdout validation |
| `bc8b7cf` | Add depth estimation comparison results |
| `48e50ed` | Add Falcon Perception adapter and KITTI evaluation |
| `ddd1aa2` | Add controlled YOLOv8x KITTI camera evaluation |
| `f63ea07` | Add YOLOv8x and Falcon camera comparison results |
| `bf3154d` | Add Falcon LiDAR single-frame fusion |
| `4142c45` | Add Falcon LiDAR multiframe and holdout evaluation |
| `fc30b04` | Add controlled YOLOv8x and Falcon LiDAR comparison |

This table should be extended after every important experimental milestone.

---

# 41. Dissertation-ready findings currently supported

The following conclusions are currently supported by the experiments already completed.

## Finding A — Simpler LiDAR association generalized better

A median of projected LiDAR depths inside a 2D detection box outperformed a more complicated nearest-cluster heuristic. A development-tuned edge-aware hybrid produced only a small development gain and failed to improve holdout MAE/RMSE.

**Dissertation implication:** complexity should be justified by independent validation rather than assumed to improve performance.

## Finding B — Projected ROI fusion has predictable failure modes

The largest depth errors are associated with mixed-depth boxes, crowding, severe truncation, occlusion and long-range LiDAR sparsity.

**Dissertation implication:** simple projection-based association is effective as a baseline but limited by 2D box geometry and sparse sensing.

## Finding C — Falcon has a distinct output failure mode

Raw Falcon can produce many redundant boxes, creating extremely low precision on some scenes despite detecting many genuine vehicles.

**Dissertation implication:** open-vocabulary generative perception can require different post-processing assumptions from conventional detectors.

## Finding D — Transparent score-free post-processing can help Falcon

IoU-cluster deduplication at threshold 0.50 reduced duplicate false positives dramatically on the development pilot and improved precision/F1 on the independent holdout at IoU 0.30 and 0.50, while introducing a small strict-IoU tradeoff.

**Dissertation implication:** post-processing must be validated independently and its tradeoffs reported.

## Finding E — YOLOv8x currently has the stronger overall fixed-threshold camera result

On the independent 10-frame camera holdout, YOLOv8x achieved higher precision and F1 across the tested IoU thresholds. Falcon + dedup achieved slightly higher recall at IoU 0.30 and 0.50.

**Dissertation implication:** the models offer different tradeoffs rather than a simple winner/loser narrative.

## Finding F — The multimodal detector comparison is split-dependent

With the LiDAR projection and depth-association procedure held fixed, YOLOv8x + LiDAR was stronger on the development split. On the independent holdout, Falcon + LiDAR produced one additional matched fused car and slightly lower MAE and RMSE, while median absolute error was nearly identical.

**Dissertation implication:** the multimodal result supports a split-dependent tradeoff rather than a universal ranking between the two camera detectors.

---

# 42. Claims that must NOT be made yet

Do not write any of the following in the final dissertation unless later experiments support them:

- “Falcon is more accurate than YOLO.”
- “YOLO is universally better than Falcon.”
- “The current camera results are official KITTI mAP.”
- “Falcon confidence threshold was X.”
- “Falcon supplies conventional per-box confidence scores in this implementation.”
- “The current timing proves YOLO is exactly X times faster end to end.”
- “V3 improves depth estimation.”
- “The full Azitt PV-RCNN++ pipeline was reproduced on the Mac.”
- “BEVFusion, PointPillars, OpenPCDet or nuScenes experiments are complete.”

---

# 43. Recommended final dissertation structure based on the work so far

## Introduction

- autonomous robotic perception problem,
- complementary properties of RGB and LiDAR,
- motivation for conventional and open-vocabulary perception,
- project objectives and research questions.

## Literature review

- 2D object detection,
- LiDAR 3D detection,
- early / late / BEV fusion,
- CLOCs,
- PointPainting,
- TransFusion / BEVFusion,
- open-vocabulary / multimodal perception,
- robustness and cross-dataset generalization.

## Methodology

- software-only architecture,
- hardware and environment separation,
- KITTI calibration and projection,
- YOLOv8x baseline,
- Falcon adapter,
- score-free Falcon deduplication,
- LiDAR median-ROI association,
- controlled evaluation splits,
- development vs holdout procedure,
- metrics and runtime protocol.

## Results

- V1/V2/V3 depth table,
- distance / occlusion / truncation analysis,
- Falcon raw vs dedup ablation,
- YOLO vs Falcon camera comparison,
- controlled YOLOv8x+LiDAR vs Falcon+LiDAR comparison,
- later 3D detector / fusion / nuScenes / robustness results.

## Discussion

- why V2 failed,
- why V3 did not generalize,
- front-surface vs object-center depth bias,
- Falcon duplicate-generation behavior,
- precision–recall tradeoff between YOLO and Falcon,
- computational tradeoffs,
- limits of simple projected ROI association,
- generalization and robustness.

## Limitations

- small current camera pilot,
- custom fixed-threshold matching is not official KITTI mAP,
- local hardware differences,
- Falcon score limitations,
- GPU-heavy experiments separated from Mac development,
- dataset and calibration assumptions.

---

# 44. Experiment logging template for future work

Every major experiment added after this point should append a section using the following structure.

```markdown
## YYYY-MM-DD — Experiment name

### Objective
What question was being tested?

### Controlled variables
What was held fixed?

### Changed variable
What was intentionally changed?

### Dataset / split
Exact frames, subset, seed, or official split.

### Configuration
Model, weights, thresholds, environment, hardware.

### Result
Exact metrics and output paths.

### Problem encountered
What failed or looked suspicious?

### Diagnosis
What evidence explained the problem?

### Fix / ablation
What was changed and why?

### Validation
Was the fix checked on independent data?

### Final decision
What implementation is retained?

### Dissertation interpretation
What can legitimately be concluded?

### Commit
Git commit hash and message.
```

---

# 45. Current immediate next action

The controlled **YOLOv8x + LiDAR vs Falcon + LiDAR** comparison is complete and preserved as a frozen experimental checkpoint.

The next technical stage should extend the project beyond the current projected-LiDAR baseline:

1. evaluate one current released YOLO-family detector if the comparison is justified by time and compatibility,
2. establish a genuine LiDAR-only 3D baseline, preferably PointPillars,
3. implement one stronger multimodal fusion method,
4. adapt the pipeline to nuScenes for cross-dataset evaluation,
5. perform targeted robustness experiments,
6. run a standardized end-to-end runtime benchmark.

The project should prioritize controlled, independently validated experiments over adding multiple overlapping frameworks without sufficient analysis.

---

# 46. Maintenance rule

This document is a **living project record**. After every significant experiment, append:

1. exact configuration,
2. exact outputs,
3. exact numerical results,
4. unexpected problems,
5. diagnosis,
6. attempted fix,
7. independent validation if relevant,
8. final design decision,
9. dissertation-safe interpretation,
10. Git commit.

This ensures that the final MSc dissertation can be reconstructed from version-controlled evidence rather than from memory alone.

---

# 47. 2026-09-19 — Controlled YOLOv8x + LiDAR vs Falcon + LiDAR

## Objective

Compare YOLOv8x + LiDAR and Falcon + LiDAR while keeping the downstream LiDAR association and evaluation procedure fixed.

## Controlled settings

- same KITTI development and holdout frame sets,
- `Car` ground-truth class,
- GT match IoU = 0.30,
- minimum projected LiDAR points = 3,
- same KITTI calibration and LiDAR projection,
- same V1 median-ROI depth estimator,
- YOLOv8x confidence threshold = 0.25,
- YOLOv8x inference on Apple MPS,
- Falcon score-free deduplication IoU = 0.50.

## YOLOv8x + LiDAR development result

| Metric | Result |
|---|---:|
| Frames | 10 |
| Ground-truth cars | 48 |
| YOLOv8x car detections | 74 |
| LiDAR-fused detections | 70 |
| Matched fused cars | 45 |
| MAE | 1.492 m |
| Median AE | 0.923 m |
| RMSE | 2.402 m |
| Mean signed error | -0.860 m |

The saved development CSV was independently re-read and all four depth-error metrics were reproduced.

## YOLOv8x + LiDAR holdout result

| Metric | Result |
|---|---:|
| Frames | 10 |
| Ground-truth cars | 45 |
| YOLOv8x car detections | 62 |
| LiDAR-fused detections | 58 |
| Matched fused cars | 41 |
| MAE | 1.789 m |
| Median AE | 1.471 m |
| RMSE | 2.355 m |
| Mean signed error | -1.029 m |

All 10 holdout frames contributed at least one matched fused car.

The saved holdout CSV was independently re-read and all four depth-error metrics were reproduced.

## Controlled YOLOv8x + LiDAR vs Falcon + LiDAR comparison

### Development split

| Metric | YOLOv8x + LiDAR | Falcon + LiDAR |
|---|---:|---:|
| Matched fused cars | 45 | 41 |
| MAE | 1.492 m | 1.783 m |
| Median AE | 0.923 m | 1.105 m |
| RMSE | 2.402 m | 2.872 m |
| Mean signed error | -0.860 m | -1.261 m |

### Independent holdout

| Metric | YOLOv8x + LiDAR | Falcon + LiDAR |
|---|---:|---:|
| Matched fused cars | 41 | 42 |
| MAE | 1.789 m | 1.655 m |
| Median AE | 1.471 m | 1.477 m |
| RMSE | 2.355 m | 2.316 m |
| Mean signed error | -1.029 m | -0.948 m |

## Implementation issue encountered

The first converted YOLOv8x evaluator passed `py_compile` but failed at runtime during ground-truth matching because the shared `iou()` function had been removed together with Falcon-specific box utilities.

The runtime error was `NameError: name 'iou' is not defined`.

The shared IoU helper was restored, the script was syntax-checked again, and the full development evaluation then completed successfully.

This was a code-conversion issue rather than a detector, dataset, calibration or LiDAR failure.

## Result artifacts

Created:

- `results/kitti/camera/lidar_depth_comparison/yolov8_falcon_lidar_comparison.csv`
- `results/kitti/camera/lidar_depth_comparison/figures/yolov8_falcon_lidar_comparison.png`

The comparison figure contains the depth-error metrics only.

Runtime was intentionally excluded from the formal comparison because the current YOLO and Falcon timing paths are not yet standardized sufficiently for a fair end-to-end speed comparison.

## Final interpretation

YOLOv8x + LiDAR was stronger on the development split, with more matched fused cars and lower MAE, Median AE and RMSE.

This advantage did not transfer in the same form to the independent holdout. Falcon + LiDAR produced one additional matched fused car and slightly lower MAE and RMSE, while the Median AE values were nearly identical.

The result is therefore treated as a split-dependent comparison rather than evidence that either detector is universally superior.

The reported depth metrics are conditional on fused detections matched to KITTI `Car` annotations at 2D IoU >= 0.30. These results are not official KITTI AP/mAP measurements.

---

# 48. 2026-09-21 — Controlled YOLOv8x vs YOLO26x Camera-LiDAR Comparison

## Objective

Evaluate whether replacing YOLOv8x with YOLO26x changes the performance of the existing camera-LiDAR depth-association pipeline while keeping the downstream fusion and evaluation procedure fixed.

## Controlled settings

The YOLO26x evaluators were produced directly from the validated YOLOv8x evaluators. Apart from the detector weights, model labels and output paths, the evaluation logic was unchanged.

The following settings were held fixed:

- same KITTI development and holdout frame sets,
- `Car` ground-truth class,
- detector confidence threshold = 0.25,
- GT matching IoU = 0.30,
- minimum projected LiDAR points = 3,
- same KITTI calibration and LiDAR projection,
- same V1 median-ROI depth estimator,
- same greedy one-to-one ground-truth matching,
- Apple MPS inference.

Environment:

- Ultralytics 8.4.132,
- PyTorch 2.13.0,
- MPS available,
- YOLO26x weights: `yolo26x.pt`.

## Development result

| Metric | YOLOv8x | YOLO26x |
|---|---:|---:|
| Ground-truth cars | 48 | 48 |
| Camera detections | 74 | 64 |
| LiDAR-fused detections | 70 | 60 |
| Matched fused cars | 45 | 44 |
| MAE | 1.492 m | 1.272 m |
| Median AE | 0.923 m | 0.875 m |
| RMSE | 2.402 m | 1.958 m |
| Mean signed error | -0.860 m | -0.623 m |

YOLO26x produced fewer camera detections and one fewer matched fused car on the development split, but the matched detections had lower MAE, median absolute error and RMSE.

## Independent holdout result

| Metric | YOLOv8x | YOLO26x |
|---|---:|---:|
| Ground-truth cars | 45 | 45 |
| Camera detections | 62 | 54 |
| LiDAR-fused detections | 58 | 53 |
| Matched fused cars | 41 | 41 |
| MAE | 1.789 m | 1.669 m |
| Median AE | 1.471 m | 1.430 m |
| RMSE | 2.355 m | 2.201 m |
| Mean signed error | -1.029 m | -0.913 m |

On the holdout split, YOLO26x retained the same number of matched fused cars as YOLOv8x while producing fewer camera detections and lower depth-error values.

## Interpretation

The same direction of depth-error reduction was observed on both development and holdout data.

However, these results should not be interpreted as a complete detector-performance ranking. The reported depth errors are conditional on fused detections that were successfully matched to KITTI `Car` ground truth at IoU >= 0.30.

The development split also shows the importance of considering match coverage alongside depth error, because YOLO26x achieved lower error while matching one fewer ground-truth car.

The result therefore indicates that the camera detector can influence downstream LiDAR depth association even when the fusion procedure itself is unchanged.

## Runtime note

The first inference in each experiment was substantially slower than subsequent frames because of model warm-up. Runtime from these runs is therefore not used as the formal detector-speed comparison.

A separate standardized runtime benchmark will be performed later.

## Result artifacts

Created:

- `results/kitti/camera/lidar_depth_comparison/yolov8_yolo26_lidar_comparison.csv`
- `results/kitti/camera/lidar_depth_comparison/figures/yolov8_yolo26_lidar_comparison.png`
- `scripts/evaluate_yolo26_lidar_multiframe.py`
- `scripts/evaluate_yolo26_lidar_holdout.py`
- `scripts/generate_yolov8_yolo26_lidar_comparison.py`

The comparison figure reports MAE, median absolute error and RMSE for both development and holdout splits.

# 49. 2026-09-22 — PointPillars LiDAR-Only 3D Baseline

## Objective

Establish a genuine LiDAR-only 3D object-detection baseline on the standard KITTI validation split before introducing a stronger multimodal fusion method.

## Model and provenance

Framework:

- MMDetection3D 1.4.0
- repository commit: `fe25f7a51d36e3702f961e198894580d83c4387b`

Configuration:

`configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py`

Official pretrained checkpoint SHA-256:

`d42d15edce05b552fdd14e6412fc1d3e02207ee4799e6e3869f61bd30e730f3e`

The model is Car-only and uses reduced KITTI LiDAR point clouds.

## Evaluation environment

- Linux
- Python 3.10.14
- PyTorch 2.1.2 + CUDA 12.1
- Tesla T4
- MMEngine 0.10.7
- MMCV 2.1.0
- MMDetection 3.2.0
- MMDetection3D 1.4.0
- NumPy 1.26.4
- Numba 0.59.1

The GPU environment is intentionally separate from the Mac YOLO/Falcon environments.

## Dataset and split

Standard SECOND KITTI split:

- training frames: 3,712
- validation frames: 3,769
- train/validation overlap: 0
- union: 7,481

Split SHA-256:

- `train.txt`: `b6417a1d9b18c8fdb085128e633d28ff321b7674a6d1b3841b8f43d865b281cb`
- `val.txt`: `6e2394d97c866c3af1ffb049f828abdf3d5b707d9575d16885fd0de87b72b0c8`

## Prediction generation

Complete validation inference produced:

- 3,769 prediction records
- 23,844 retained detections
- 127 frames with zero retained predictions
- 0 frames with non-finite values

Frozen prediction PKL SHA-256:

`2905f9c15fd04a6055dca99753b6fb6e6585492371608c5a625e3cc6ded764d3`

The bulk prediction PKL remains outside Git.

## Evaluation issue and resolution

The original MMDetection3D run completed inference for all 3,769 validation frames and converted all predictions to KITTI format, but terminated with exit status 139 during BEV/3D evaluation.

The failure was isolated to the legacy Numba CUDA rotated-IoU path.

A minimal identical-box rotated-IoU test reproduced the native crash. The environment was then given `cuda-python==12.2.1` and evaluation was invoked with:

`NUMBA_CUDA_USE_NVIDIA_BINDING=1`

The same identical-box test then returned IoU = 1.0 successfully.

A 50-frame bbox/BEV/3D preflight completed before the full frozen prediction set was evaluated.

No PointPillars inference was repeated for the final metrics.

## Official KITTI result

Primary protocol:

**KITTI Car AP40, strict IoU = 0.70**

| Metric | Easy | Moderate | Hard |
|---|---:|---:|---:|
| 2D AP40 | 96.1046 | 92.4841 | 89.7698 |
| BEV AP40 | 92.7663 | 88.3680 | 85.5054 |
| 3D AP40 | **90.3449** | **79.1299** | **76.1566** |

Primary Moderate 3D result:

**79.1299% AP40 at IoU 0.70.**

## Independent numerical verification

The full KITTI evaluation was rerun directly from the frozen predictions.

Verification:

- prediction SHA-256 unchanged
- saved metrics: 36
- recomputed metrics: 36
- maximum absolute difference: 0.0
- every metric exactly identical

## Earlier custom-frame overlap audit

The earlier camera/fusion pilot used 20 custom development/holdout frames.

Relative to the standard SECOND split:

- 12 are standard-validation frames
- 8 are standard-training frames

Only the 12 validation frames are eligible for independent frame-matched PointPillars comparisons.

## Moderate-Car localization diagnostic

This object-level diagnostic is not official KITTI recall/AP.

Valid Moderate Car GT objects: 7,874.

Best retained 3D-IoU distribution:

- IoU >= 0.70: 6,644
- 0.50 <= IoU < 0.70: 934
- 0.30 <= IoU < 0.50: 50
- 0 < IoU < 0.30: 18
- no overlap: 228

Distance-stratified strict-IoU coverage:

| Distance | Valid GT | Best prediction IoU >= 0.70 |
|---|---:|---:|
| <20 m | 3,088 | 96.11% |
| 20–40 m | 3,793 | 83.63% |
| 40–60 m | 993 | 50.76% |

This supports a clear distance-dependent degradation in strict 3D localization.

## Prediction-centric diagnostic

This diagnostic is not the official KITTI false-positive calculation.

For score >= 0.50 unmatched predictions:

- 3,678 strongly overlapped Moderate-ignored GT
- 657 moderately overlapped Moderate-ignored GT
- 874 were valid-Car localization near-misses
- 34 had weak overlap with valid Car GT
- 66 had low overlap
- 1,082 had no 3D overlap with evaluated or Moderate-ignored GT

This refinement prevents ignored-object detections from being naively counted as ordinary false positives.

## Frozen qualitative cases

Cases were selected by quantitative rules before visualization.

### Distant success

Frame `002744`:

- distance: 40.70 m
- prediction score: 0.896138
- 3D IoU: 0.913522

### Distant localization near-miss

Frame `004096`:

- distance: 40.45 m
- prediction score: 0.958086
- 3D IoU: 0.680505

### High-confidence unmatched diagnostic

Frame `001199`:

- prediction depth: 33.30 m
- score: 0.964747
- best valid Moderate GT IoU: 0.0
- best ignored Moderate GT IoU: 0.0

This case is described as a diagnostic unmatched prediction rather than automatically as an official KITTI false positive.

## Supported interpretation

PointPillars provides a strong LiDAR-only 3D baseline on KITTI while showing clear degradation in strict 3D localization as object distance increases.

Many strict-threshold failures are localization near-misses rather than complete detection failures.

The analysis also shows why apparent unmatched predictions must be interpreted using KITTI ignored-object rules.

## Limitations

- Distance coverage is diagnostic, not official KITTI AP/recall.
- Prediction-centric unmatched analysis does not replicate every part of the official KITTI false-positive protocol.
- T4 timing is not directly comparable with Mac YOLO/Falcon runtime.
- Eight earlier custom frames belong to the standard training split.
- The experiment reproduces an official pretrained PointPillars checkpoint rather than training a new project-specific model.

## Result artifacts

Compact GitHub artifacts include:

- `results/kitti/pointpillars/pointpillars_kitti_ap40_strict_summary.csv`
- `results/kitti/pointpillars/figures/pointpillars_kitti_ap40_strict.png`
- `results/kitti/pointpillars/figures/pointpillars_kitti_ap40_strict.pdf`
- `results/kitti/pointpillars/kitti_full_metrics.json`
- `results/kitti/pointpillars/kitti_full_results.txt`
- `results/kitti/pointpillars/figures/pointpillars_distance_diagnostic.png`
- `results/kitti/pointpillars/figures/pointpillars_distance_diagnostic.pdf`
- `results/audits/kitti/kitti_split_overlap_audit.json`
- `results/kitti/pointpillars/pointpillars_evidence_manifest.json`
- `results/kitti/pointpillars/pointpillars_moderate_failure_summary.json`
- `results/kitti/pointpillars/pointpillars_prediction_diagnostics_summary.json`
- `results/kitti/pointpillars/pointpillars_high_conf_unmatched_refined.json`
- `results/kitti/pointpillars/pointpillars_final_case_studies.json`
- three frozen qualitative case-study figures in PNG and PDF form

Bulk predictions, model weights and datasets remain outside Git history.

## Final decision

The pinned MMDetection3D PointPillars Car checkpoint is retained as the project's principal LiDAR-only 3D baseline.

The next major experimental stage is one stronger multimodal fusion method while preserving PointPillars as the LiDAR reference.

## Commit

Local repository transfer and integrity verification completed. Milestone commit message: `Add PointPillars LiDAR baseline and KITTI evaluation`.

---

## 2026-09-25 — Full KITTI CLOCs GT audit and CPU 3D-IoU reference

The canonical SECOND train/validation manifest was verified with 3,712 training and 3,769 validation frames and no overlap. The Mac audit `scripts/audit_kitti_gt.py` processed all 7,481 KITTI labelled frames and saved `results/kitti/clocs/training_exports/kitti_clocs_gt_audit.json` (commit `96c3a37`). For the exact Car class, 28,742 GT boxes precede the pinned CLOCs/SECOND BEV range filter; 28,374 remain, with 368 filtered out. Training retains 14,144 Cars and validation retains 14,230. There are 859 frames with no Car GT after filtering; these can legitimately yield all-negative candidate targets.

The GT audit and 84-test Mac suite passed at `96c3a37`. The saved `000001_dataset_schema_compatibility.npz` is a structural smoke test only: its 107,136 candidates all have placeholder zero IoU and negative labels. It must not be used to train or evaluate a fusion model.

Added a NumPy camera-space rotated 3D-IoU reference with validated synthetic overlap, rotation, height, empty-GT and symmetry cases. The dense PointPillars adapter now returns aligned camera boxes in addition to LiDAR boxes and features. Seven new tests passed locally; the available repository subset ran 43 tests, all passing. This is code-level validation only: the Mac currently lacks dense saved 3D candidate boxes, so no real candidate-to-GT targets or fusion AP have been measured. Once Colab is available, export frozen dense boxes with scores and indices, compute real multi-frame IoUs, independently cross-check against the pinned CLOCs evaluation kernel, inspect positive/negative/ignored counts, then permit training.

This is an engineering record toward the 8–10 October Phase 1 freeze; dissertation chapter writing starts after 10 October.

### Target-construction command

Implemented `scripts/build_clocs_training_example.py` with a frame-tagged dense-box/feature export contract (`docs/CLOCS_TARGET_PREPARATION.md`). It rejects mismatched frames, altered dense candidate order, mismatched feature scores or distances, missing frozen-split membership, and malformed 3D geometry. The complete synthetic CLI integration test saves and reloads a hashed example containing positive, ignored and negative targets. The available local suite passed 49/49; the Mac full suite and genuine dense export remain to be checked after pulling and restoring Colab.

### 2026-09-25 Mac confirmation and 2D alignment guard

The Mac pulled commit `e2e3a47` and passed the complete repository suite: **97 tests in 0.613 s**, with a clean `main...origin/main` working tree. This confirms the CPU geometry and first target-construction version run in `azitt310` on macOS.

The input contract was then tightened: saved original 2D boxes, scores, projected candidate boxes, 2D detector indices and row types are required to reconstruct **every** CLOCs feature row. A wrong 2D export or candidate association can no longer pass using only matching 3D scores and distances. Local tests cover altered 2D detections and projections; the available subset passed 52 tests. This remains a code-level check; real candidate-to-GT labels still require the Colab export.

### 2026-09-25 — Daily engineering target and milestone policy

The Mac confirmed commit `6b913a1`: 100/100 tests passed in 0.551 seconds and the repository was clean. The dated engineering schedule in `docs/PHASE1_DAILY_TARGETS_2026-09-25_TO_10-10.md` retains the full agreed technical scope through the 8–10 October freeze. Each daily exit criterion requires a measured artifact or an explicit blocker; small code, manifests, tests, results and evidence are committed to GitHub by milestone. Large datasets, predictions and checkpoint files stay in durable project storage with hashes and locations tracked in the repository. GPU-dependent experiments remain conditional on access to compute; dissertation chapter writing begins after 10 October.

### 2026-09-25 — Reconciled locked 29 August scope

The user supplied the earlier locked-plan screenshot and reaffirmed MMDetection3D, the Azitt fusion baseline, Falcon, OpenPCDet, BEVFusion and MultiCorrupt; YOLOv8 reproduction and a modern YOLO baseline; PV-RCNN++/PointPillars; KITTI and nuScenes; semantic fusion, robustness, ablations, runtime and a Glasgow demo. NVIDIA-specific optimisation was designated **stretch only**. The daily schedule was amended to name each method explicitly and require its own measured evidence or exact blocker. Existing projected-LiDAR, YOLO, Falcon and PointPillars results remain separate from unverified Azitt/PV-RCNN++/BEVFusion reproductions. The old screenshot's Mac environment setup is already complete; real aligned dense 3D candidates remain the next gate. Previously generated figures will be inventoried and preserved before new images are produced from verified runs.

## 2026-10-28 — synthesis of illumination/weather cases and arm feedback

The [supervisor follow-up](results/supervisor_followup/20261028/README.md) consolidates completed A–D darkening, ten-class and original-visibility diagnostics, rainy examples and the two preselected night frames. The [discussion draft](docs/report/ILLUMINATION_WEATHER_DISCUSSION_2026-10-28.md) records measurement, competing explanations, limitations and implications. It retains the distinction between controlled synthetic input changes, observational rain/night cases and representative weather evaluation.

A new arm-clarification composite uses the preserved profile parameters and all 45 trial records, showing explicit failed placements and contact loading. Light/heavy geometry and commanded paths are shared; mass and friction both change. No camera–LiDAR-controlled manipulation or mass-only causal effect is claimed. The separate mobile-robot corridor analysis remains the main measured perception-to-alert application.

Local verification checks 48 A–D threshold rows, 12 night rows, nine reaggregated arm groups and paired offsets. No model inference, official evaluator or simulator rerun was performed. Public-file preservation code is prepared, but the mounted-Drive copy must still be run and verified. This milestone completes the bounded synthesis/clarification, not the entire dissertation or a comprehensive weather benchmark.

## 2026-10-28 — final bounded night/rain pedestrian addition

The annotation-based amendment selected scene-1067 independently of model outcomes after the two alphabetically earlier candidate scenes had no eligible pedestrian frame. The [additional case](results/nuscenes/bevfusion/case_analysis/20261028/night_rain_pedestrian/README.md) is now analysed: an eligible pedestrian at 30.06 m with one current-scan LiDAR point is missed by both pipelines at 0.10/0.30/0.50, despite nearby same-class candidates. Exact scores, frame counts, assignments, point-support checks and original-image comparison are preserved. The result closes the specific positive-labelled-frame gap, not representative night/rain evaluation or weather causality. The original two night cases remain intact; all nine unique selected frames can enter the thematic dissertation discussion. Inputs are verified on Drive; a later derived-output copy requires its own receipt.
