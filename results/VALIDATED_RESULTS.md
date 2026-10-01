# Validated Experimental Results

> **Current CLOCs status (1 October 2026):** Three aligned epoch-12 seeds completed full 3,769-frame validation with exact baseline parity. Moderate 3D AP40 = 79.8616 / 76.2666 / 78.7616, against PointPillars 79.1299. Only one seed improves Moderate; all three reduce Hard. Seed 20260928's epoch-24 continuation improved development candidate AP but reduced Moderate detection AP40 to 79.4874. Report descriptive seed variability, not statistical significance. This current diagnostic qualifies the earlier single-checkpoint results preserved below. See [findings](clocs_seed_diagnostics_20261001/FINDINGS.md) and [all metrics](clocs_seed_diagnostics_20261001/ap40_comparison.csv).

> **PV-RCNN update (30 September 2026):** Official pretrained OpenPCDet PV-RCNN achieved Car strict IoU 0.70 3D AP40 **92.1047 / 84.3604 / 82.4830** on the same 3,769-frame validation split. Native and pinned MMDetection3D evaluation agreed exactly across all nine metrics. This is a LiDAR-only reproduction, not PV-RCNN++/Azitt or an equal-training-budget ablation. See [findings and comparison](pv_rcnn_kitti/FINDINGS.md).

> **Historical CLOCs checkpoint (28–29 September 2026):** Baseline-matched reduced-FOV validation is complete on 3,769 frames. All nine historical PointPillars AP40 metrics reproduce exactly (maximum error 0.0). Car 3D Moderate is 79.1299 → 79.2542 (+0.1243 points); Hard is 76.1566 → 73.6326 (-2.5239). This mixed result supersedes the earlier raw-cloud diagnostic for the historical-baseline comparison. See [verified findings and limitations](../docs/CLOCS_BASELINE_MATCHED_RESULT_2026-09-29.md).

> 
>
> Project: **A Software-Only Deep Learning Pipeline for Multi-Modal Object Detection and Robotic Perception Using LiDAR and Camera Fusion Across Multiple Datasets**
>
> This file consolidates the main numerical results produced so far. It is intended for supervisor/professor review and later dissertation use. Raw datasets, model weights, and bulk model outputs are intentionally not version-controlled.

## 1. Evaluation conventions and important caveats

- Primary dataset so far: **KITTI training set**.
- Camera class evaluated: **Car**.
- Exploratory depth-association matching uses **2D IoU >= 0.30** with greedy one-to-one assignment.
- Falcon downstream output uses the frozen **score-free IoU deduplication threshold 0.50**.
- Projected-LiDAR depth baseline uses:
  - projected KITTI Velodyne points,
  - points inside the 2D camera box,
  - at least **3** LiDAR returns,
  - median camera-coordinate depth.
- Depth MAE / Median AE / RMSE are **conditional on camera detections that were successfully matched to KITTI Car annotations**. They are not full object-detection accuracy metrics.
- The camera comparison is a **controlled fixed-operating-point pilot**, not official KITTI AP/mAP.
- Runtime numbers below are not yet a fully standardized end-to-end benchmark unless explicitly stated.

---

# 2. Projected LiDAR depth-estimation study

## 2.1 Single-frame diagnostic — KITTI 000149

9 YOLOv8x detections were matched to KITTI Car ground truth.

| Method | MAE |
|---|---:|
| V1 Median ROI | **1.825 m** |
| V2 Nearest Cluster | 7.141 m |

V1 was substantially better on this frame.

## 2.2 Multi-frame development progression

### 10-frame development result

45 matched detections.

| Method | MAE | Median AE | RMSE |
|---|---:|---:|---:|
| V1 Median ROI | **1.492 m** | **0.923 m** | **2.402 m** |
| V2 Nearest Cluster | 2.930 m | 1.409 m | 4.930 m |

### 50-frame development result

180 matched detections.

| Method | MAE | Median AE | RMSE |
|---|---:|---:|---:|
| V1 Median ROI | **1.577 m** | **1.098 m** | **2.545 m** |
| V2 Nearest Cluster | 3.423 m | 1.451 m | 6.368 m |

### 100-frame development result

| Method | N | MAE | Median AE | RMSE |
|---|---:|---:|---:|---:|
| V1 Median ROI | 360 | 1.897845 | 1.264696 | 3.750320 |
| V2 Nearest Cluster | 359 | 3.985658 | 1.684138 | 6.674903 |
| V3 Edge-Aware Hybrid | 360 | **1.861833** | **1.264696** | **3.655738** |

V3 appeared slightly better on the development set, therefore it was tested independently before any design change was accepted.

## 2.3 Independent 100-frame holdout

A deterministic holdout was generated with random seed 42 while excluding the development frames.

| Method | N | MAE | Median AE | RMSE |
|---|---:|---:|---:|---:|
| V1 Median ROI | 351 | **1.953883** | 1.317563 | **3.924869** |
| V2 Nearest Cluster | 350 | 4.194522 | 1.631340 | 6.957196 |
| V3 Edge-Aware Hybrid | 351 | 1.961820 | **1.284635** | 3.947876 |

### Frozen decision

**V1 Median ROI was retained.**

V3 did not improve holdout MAE or RMSE, so the small development gain was not treated as sufficient evidence to replace V1.

Tracked artifacts:

- [depth_method_comparison.csv](depth_method_comparison.csv)
- [depth_method_comparison.png](depth_method_comparison.png)

## 2.4 Signed depth bias

On the 100-frame development evaluation:

- V1 mean signed error: **-1.075 m**
- V2 mean signed error: **-3.674 m**

The negative bias is physically plausible because projected LiDAR returns commonly lie on the visible/front surface of a vehicle, while the KITTI object location z is the annotated 3D object location in camera coordinates.

## 2.5 V1 distance analysis

| GT distance | N | MAE | Median AE | RMSE |
|---|---:|---:|---:|---:|
| 0-20 m | 143 | 1.474 | 0.780 | 2.674 |
| 20-40 m | 144 | 1.878 | 1.407 | 2.999 |
| 40-60 m | 59 | 2.176 | 1.713 | 3.446 |
| 60+ m | 14 | 5.258 | 2.013 | 12.087 |

The simple projected-ROI method degrades at long range, where LiDAR support is sparse and association errors become more damaging.

## 2.6 Visibility / occlusion analysis for V1

### Truncation

| Truncation | N | MAE | Median AE | RMSE |
|---|---:|---:|---:|---:|
| 0-0.15 | 306 | 1.834 | 1.302 | 3.707 |
| >0.15-0.30 | 6 | 0.181 | 0.197 | 0.208 |
| >0.30-0.50 | 9 | 0.632 | 0.527 | 0.750 |
| >0.50 | 39 | 2.959 | 1.184 | 4.680 |

The middle truncation bins are too small for strong general conclusions.

### Occlusion

| Occlusion | N | MAE | Median AE | RMSE |
|---|---:|---:|---:|---:|
| 0 - fully visible | 194 | 1.603 | 1.347 | 2.455 |
| 1 - partly occluded | 97 | 1.676 | 0.904 | 4.807 |
| 2 - largely occluded | 59 | 3.504 | 1.653 | 5.290 |
| 3 - unknown | 10 | 0.296 | 0.217 | 0.367 |

Occlusion level 2 was associated with materially larger errors.

---

# 3. Falcon Perception camera evaluation

## 3.1 Single-frame diagnostic — KITTI 000149

- Falcon boxes: **11**
- KITTI Car GT boxes: **10**

| IoU threshold | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| 0.30 | 10 | 1 | 0 | 0.909 | 1.000 |
| 0.50 | 7 | 4 | 3 | 0.636 | 0.700 |
| 0.70 | 4 | 7 | 6 | 0.364 | 0.400 |

This is a diagnostic only, not a final accuracy claim.

## 3.2 Raw Falcon — 10-frame development split

Development frames:

`000000, 000831, 001662, 002493, 003324, 004155, 004986, 005817, 006648, 007480`

- KITTI Car GT: **48**
- Raw Falcon boxes: **168**

| IoU | TP | FP | FN | Precision | Recall | F1 | Mean matched IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.30 | 41 | 127 | 7 | 0.244 | 0.854 | 0.380 | 0.678 |
| 0.50 | 31 | 137 | 17 | 0.185 | 0.646 | 0.287 | 0.751 |
| 0.70 | 21 | 147 | 27 | 0.125 | 0.438 | 0.194 | 0.821 |

Mean generation time: **6.416 s/frame**, approximately **0.156 generation FPS**.

## 3.3 Duplicate failure case — frame 006648

- KITTI GT cars: **9**
- Raw Falcon boxes: **101**
- Falcon generation time: **25.97 s**

Visual inspection showed heavy redundant/overlapping output rather than simply complete object-detection failure.

## 3.4 Falcon + score-free dedup 0.50 — development

The project uses connected-component IoU clustering and a coordinate-wise median representative because the Falcon output path used here does not provide conventional per-box confidence scores.

| Eval IoU | TP | FP | FN | Precision | Recall | F1 | Mean IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.30 | 41 | 37 | 7 | 0.526 | 0.854 | 0.651 | 0.677 |
| 0.50 | 31 | 47 | 17 | 0.397 | 0.646 | 0.492 | 0.749 |
| 0.70 | 21 | 57 | 27 | 0.269 | 0.438 | 0.333 | 0.821 |

Frame 006648 was reduced from **101 raw boxes to 17 deduplicated boxes**.

## 3.5 Independent Falcon camera holdout

Holdout frames:

`000642, 000666, 000704, 001506, 003236, 003280, 004891, 005215, 005783, 006419`

KITTI Car GT: **45**.

### Raw Falcon

| IoU | TP | FP | FN | Precision | Recall | F1 | Mean matched IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.30 | 42 | 33 | 3 | 0.560 | 0.933 | 0.700 | 0.754 |
| 0.50 | 38 | 37 | 7 | 0.507 | 0.844 | 0.633 | 0.790 |
| 0.70 | 30 | 45 | 15 | 0.400 | 0.667 | 0.500 | 0.836 |

### Falcon + dedup 0.50

| IoU | TP | FP | FN | Precision | Recall | F1 | Mean matched IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.30 | 42 | 30 | 3 | 0.583 | 0.933 | 0.718 | 0.751 |
| 0.50 | 38 | 34 | 7 | 0.528 | 0.844 | 0.650 | 0.787 |
| 0.70 | 29 | 43 | 16 | 0.403 | 0.644 | 0.496 | 0.840 |

### Frozen decision

Falcon downstream experiments use **dedup IoU = 0.50**, while raw Falcon remains an ablation baseline.

---

# 4. Controlled YOLOv8x camera evaluation

The same development and independent holdout frame sets were used.

YOLOv8x operating point:

- confidence threshold: **0.25**
- Apple MPS backend
- KITTI Car evaluation
- same greedy one-to-one matching at IoU 0.30 / 0.50 / 0.70

## 4.1 Development

| IoU | TP | FP | FN | Precision | Recall | F1 | Mean matched IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.30 | 45 | 29 | 3 | 0.608 | 0.938 | 0.738 | 0.710 |
| 0.50 | 40 | 34 | 8 | 0.541 | 0.833 | 0.656 | 0.746 |
| 0.70 | 24 | 50 | 24 | 0.324 | 0.500 | 0.393 | 0.838 |

Development mean timing included first-frame warm-up and is therefore not used as a representative steady-state value.

## 4.2 Holdout

| IoU | TP | FP | FN | Precision | Recall | F1 | Mean matched IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.30 | 41 | 21 | 4 | 0.661 | 0.911 | 0.766 | 0.767 |
| 0.50 | 36 | 26 | 9 | 0.581 | 0.800 | 0.673 | 0.815 |
| 0.70 | 30 | 32 | 15 | 0.484 | 0.667 | 0.561 | 0.856 |

Holdout mean prediction time: **0.058 s/frame**, approximately **17.216 FPS**.

Tracked artifacts:

- [camera_detector_comparison.csv](camera_detector_comparison.csv)
- [camera_detector_comparison.png](camera_detector_comparison.png)

## 4.3 Controlled camera interpretation

On the independent holdout:

- YOLOv8x produced higher precision and F1 across the tested IoU thresholds.
- Falcon + dedup produced slightly higher recall at IoU 0.30 and 0.50.
- Falcon provides open-vocabulary flexibility but required explicit score-free duplicate handling and was materially slower in the current local setup.

The correct interpretation is a **tradeoff**, not a claim that one model is universally superior.

---

# 5. Falcon + LiDAR fusion

## 5.1 Single-frame integration — KITTI 000149

Configuration:

- Falcon query: `car`
- dedup IoU: **0.50**
- minimum LiDAR points: **3**
- depth estimator: **median projected LiDAR camera-coordinate depth inside each deduplicated box**

Result:

- Raw Falcon boxes: **11**
- Deduplicated boxes: **10**
- Fused detections: **10**

Per-detection estimated depths:

| Detection | Estimated depth | LiDAR points |
|---:|---:|---:|
| 0 | 59.97 m | 7 |
| 1 | 49.60 m | 21 |
| 2 | 41.58 m | 31 |
| 3 | 47.37 m | 66 |
| 4 | 47.44 m | 102 |
| 5 | 27.71 m | 92 |
| 6 | 27.85 m | 129 |
| 7 | 15.73 m | 309 |
| 8 | 13.21 m | 767 |
| 9 | 9.22 m | 1629 |

This was used as a functional integration check before multi-frame evaluation.

Relevant commit:

- `bf3154d` — Add Falcon LiDAR single-frame fusion

## 5.2 10-frame development evaluation

Same Falcon development frames as the earlier camera pilot.

Frozen configuration:

- dedup IoU = **0.50**
- GT match IoU = **0.30**
- minimum projected LiDAR points = **3**
- V1 median-ROI depth estimator

Aggregate result:

| Quantity | Value |
|---|---:|
| Frames evaluated | 10 |
| Ground-truth cars | 48 |
| Raw Falcon boxes | 168 |
| Deduplicated boxes | 78 |
| LiDAR-fused detections | 65 |
| Matched fused cars | 41 |
| MAE | **1.783 m** |
| Median AE | **1.105 m** |
| RMSE | **2.872 m** |
| Mean signed error | **-1.261 m** |
| Mean Falcon generation | 5.579 s/frame |
| Approx. generation FPS | 0.179 |

The saved CSV was independently re-read and the metrics were recomputed from the stored signed errors, reproducing the same values.

## 5.3 Independent 10-frame holdout evaluation

Same independent Falcon/YOLO holdout frames as the earlier camera comparison.

Aggregate result:

| Quantity | Value |
|---|---:|
| Frames evaluated | 10 |
| Ground-truth cars | 45 |
| Raw Falcon boxes | 75 |
| Deduplicated boxes | 72 |
| LiDAR-fused detections | 63 |
| Matched fused cars | 42 |
| MAE | **1.655 m** |
| Median AE | **1.477 m** |
| RMSE | **2.316 m** |
| Mean signed error | **-0.948 m** |
| Mean Falcon generation | 2.488 s/frame |
| Approx. generation FPS | 0.402 |

The holdout CSV was independently re-read and the metrics were recomputed using NumPy, reproducing:

- MAE: **1.655 m**
- Median AE: **1.477 m**
- RMSE: **2.316 m**
- Mean signed error: **-0.948 m**

### Development vs holdout interpretation

| Metric | Development | Holdout |
|---|---:|---:|
| Matched fused cars | 41 | 42 |
| MAE | 1.783 m | **1.655 m** |
| Median AE | **1.105 m** | 1.477 m |
| RMSE | 2.872 m | **2.316 m** |
| Mean signed error | -1.261 m | **-0.948 m** |

The frozen fusion method did **not collapse on the independent holdout**. Holdout MAE and RMSE were lower, while Median AE was higher. The negative depth bias remained but was smaller.

This supports a cautious statement that the simple Falcon+LiDAR depth pipeline showed **stable performance across these controlled development and holdout subsets**. It does not establish universal KITTI performance.

Relevant scripts:

- `scripts/fuse_falcon_lidar.py`
- `scripts/evaluate_falcon_lidar_multiframe.py`
- `scripts/evaluate_falcon_lidar_holdout.py`

Relevant commit:

- `4142c45` — **Add Falcon LiDAR multiframe and holdout evaluation**

## 5.4 Controlled YOLOv8x + LiDAR vs Falcon + LiDAR comparison

The final controlled experiment at this stage changes the camera detector while keeping the downstream LiDAR procedure fixed.

Fixed settings:

- same KITTI development and holdout frame sets,
- `Car` class,
- GT matching IoU = **0.30**,
- minimum projected LiDAR points = **3**,
- same KITTI calibration and LiDAR projection,
- same V1 median-ROI depth estimator,
- YOLOv8x confidence threshold = **0.25**,
- Falcon score-free deduplication IoU = **0.50**.

### Development split

| Metric | YOLOv8x + LiDAR | Falcon + LiDAR |
|---|---:|---:|
| Ground-truth cars | 48 | 48 |
| Camera boxes used for fusion | 74 | 78 |
| LiDAR-fused detections | 70 | 65 |
| Matched fused cars | **45** | 41 |
| MAE | **1.492 m** | 1.783 m |
| Median AE | **0.923 m** | 1.105 m |
| RMSE | **2.402 m** | 2.872 m |
| Mean signed error | **-0.860 m** | -1.261 m |

### Independent holdout

| Metric | YOLOv8x + LiDAR | Falcon + LiDAR |
|---|---:|---:|
| Ground-truth cars | 45 | 45 |
| Camera boxes used for fusion | 62 | 72 |
| LiDAR-fused detections | 58 | 63 |
| Matched fused cars | 41 | **42** |
| MAE | 1.789 m | **1.655 m** |
| Median AE | **1.471 m** | 1.477 m |
| RMSE | 2.355 m | **2.316 m** |
| Mean signed error | -1.029 m | **-0.948 m** |

Both development and holdout CSV files were independently re-read and their MAE, median absolute error, RMSE and mean signed error were recomputed before these results were frozen.

### Interpretation

YOLOv8x + LiDAR produced the stronger result on the development split, with more matched fused cars and lower depth-error metrics.

That advantage did not carry over in the same way to the independent holdout. Falcon + LiDAR produced one more matched fused car and slightly lower MAE and RMSE, while the two median absolute errors were almost identical.

The result is therefore treated as a **split-dependent comparison rather than evidence that either detector is universally superior**. The depth metrics remain conditional on fused detections matched to KITTI `Car` annotations at 2D IoU >= 0.30.

Runtime is not included in the formal comparison artifact because the present YOLO and Falcon timing paths are not yet sufficiently standardized for an end-to-end speed comparison.

Tracked artifacts:

- [yolov8_falcon_lidar_comparison.csv](yolov8_falcon_lidar_comparison.csv)
- [yolov8_falcon_lidar_comparison.png](yolov8_falcon_lidar_comparison.png)

Relevant scripts:

- `scripts/evaluate_yolov8_lidar_multiframe.py`
- `scripts/evaluate_yolov8_lidar_holdout.py`
- `scripts/evaluate_falcon_lidar_multiframe.py`
- `scripts/evaluate_falcon_lidar_holdout.py`

---

# 6. Current experimentally supported findings

1. **The simpler V1 median-ROI LiDAR depth estimator generalized better than the more complicated alternatives tested.**
2. **Projected ROI fusion has identifiable failure modes involving mixed depth layers, truncation, occlusion, crowding, and long-range LiDAR sparsity.**
3. **Raw Falcon has a distinct duplicate/redundancy failure mode on some scenes.**
4. **Transparent score-free deduplication substantially improves Falcon precision/F1 at moderate evaluation IoU thresholds while introducing a small strict-IoU tradeoff.**
5. **YOLOv8x currently has the stronger overall fixed-threshold camera result, while Falcon + dedup has slightly higher recall at some thresholds and offers open-vocabulary flexibility.**
6. **The frozen Falcon+LiDAR median-depth pipeline remained stable on the independent 10-frame holdout and reproduced its saved metrics on an independent CSV integrity check.**
7. **The controlled YOLOv8x+LiDAR vs Falcon+LiDAR result is split-dependent: YOLOv8x was stronger on development, while Falcon produced one additional matched fused car and slightly lower MAE/RMSE on the independent holdout.**
8. **Baseline-matched CLOCs validation on all 3,769 frames produced mixed effects: +0.1243 Moderate and -2.5239 Hard Car 3D AP40 points. The historical PointPillars baseline reproduced exactly across all nine metrics.**

---

# 7. Claims not supported yet

The following should **not** be claimed from the current evidence:

- official KITTI test-server AP, statistically significant CLOCs gains, or universal fusion superiority,
- universal superiority of YOLO over Falcon or Falcon over YOLO,
- a standardized end-to-end runtime ratio,
- completion of the full Azitt PV-RCNN++ pipeline on macOS,
- completed OpenPCDet or BEVFusion fusion results,
- completed nuScenes cross-dataset results,
- completed robustness evaluation.

---

# 8. Repository evidence map

For a professor or reviewer, the most useful files are:

- `PROJECT_HISTORY.md` — detailed chronological technical record, decisions, failures and interpretations.
- `DAILY_LOG.md` — dated working log.
- `results/VALIDATED_RESULTS.md` — this consolidated numerical summary.
- `results/depth_method_comparison.csv` — frozen V1/V2/V3 depth comparison.
- `results/depth_method_comparison.png` — figure for the depth-method comparison.
- `results/camera_detector_comparison.csv` — frozen YOLOv8x vs Falcon camera pilot.
- `results/camera_detector_comparison.png` — figure for the camera comparison.
- `results/yolov8_falcon_lidar_comparison.csv` — frozen controlled multimodal detector comparison.
- `results/yolov8_falcon_lidar_comparison.png` — figure for the controlled multimodal comparison.
- `scripts/evaluate_yolov8_lidar_multiframe.py` — YOLOv8x+LiDAR development evaluator.
- `scripts/evaluate_yolov8_lidar_holdout.py` — YOLOv8x+LiDAR independent holdout evaluator.
- `scripts/evaluate_falcon_lidar_multiframe.py` — Falcon+LiDAR development evaluator.
- `scripts/evaluate_falcon_lidar_holdout.py` — Falcon+LiDAR independent holdout evaluator.
- `results/clocs_full_val_paired/paired_diagnostic.json` — full same-run AP40 measurements and historical baseline discrepancy.
- `results/clocs_full_val_paired/coverage_diagnostic.json` — all-labelled-Car best-IoU coverage by range and occlusion, not AP40.
- `results/clocs_full_val_paired/cases/case_studies.json` — audited rescue/loss examples.
- `docs/CLOCS_FULL_VALIDATION_PAIRED_RESULT_2026-09-29.md` — interpretation and limitations.

Bulk generated outputs and datasets are intentionally excluded from Git because of size and reproducibility policy.

---

# 9. Reproducibility status

The repository records:

- exact controlled frame lists,
- thresholds,
- model roles,
- calibration procedure,
- matching rules,
- depth metric definitions,
- development/holdout separation,
- scripts used for the Falcon+LiDAR and YOLOv8x+LiDAR controlled experiments,
- version-controlled frozen result tables and figures for the depth, camera-detector and multimodal comparison studies.

The 128-frame trained YOLO26x+CLOCs checkpoint has now completed reduced-FOV validation on all 3,769 frames with exact aggregate historical PointPillars AP parity. The accepted results are in `results/clocs_full_val_reduced/`. Preserve `results/clocs_full_val_paired/` as the earlier raw-cloud diagnostic, including its mismatch; its figures and case studies must not be presented as corrected-run evidence.

---

# 10. PointPillars baseline and CLOCs target preparation

The pinned PointPillars LiDAR-only baseline was independently verified on the frozen SECOND KITTI validation split. At strict Car IoU 0.70, **3D AP40 = 90.3449 / 79.1299 / 76.1566** for Easy / Moderate / Hard. These are LiDAR-only detection metrics, not CLOCs fusion results. See `results/pointpillars_kitti_ap40_strict_summary.csv` and `results/pointpillars_evidence_manifest.json` for experiment details.

The full CLOCs GT audit processed **7,481** KITTI labelled frames (train **3,712**, validation **3,769**, overlap **0**). Exact Car GT counts were **28,742 before** and **28,374 after** the detector range filter, with **14,144** retained in training and **14,230** in validation. **859** frames have no retained Car GT. Source: `results/kitti_clocs_gt_audit.json` (commit `96c3a37`). The audit verifies GT preparation and split integrity; it is not an evaluation of CLOCs accuracy.

Aligned real PointPillars dense candidates are available for two canonical training frames. Frame `000000` is a valid empty-Car/empty-YOLO negative control; frame `000149` has 10 retained Car GT, 9 YOLO Car detections, 107,136 candidate anchors, 158,838 feature rows, and 179 positive / 106,798 negative / 159 ignored candidates. On `000149`, a hash-verified 258-anchor stratified CPU comparison against independent SciPy/Qhull geometry passed with zero label disagreements and maximum absolute IoU error `2.968672907499581e-08` versus tolerance `0.0005`. This verifies a single real-frame geometry implementation, **not** the pinned CUDA evaluator, trained CLOCs, or KITTI AP. The CUDA reference crashed with signal 11 before producing a result. The earlier all-negative `reference_structural_smoke` uses placeholder zero IoU and remains excluded from training. Sources: `results/clocs_real_frame_000000_audit.json`, `results/clocs_real_frame_000149_audit.json`, and `results/clocs_000149_iou_scipy_audit.json`.


---

# 11. Additional controlled camera-model pilot

A tracked YOLOv8x versus YOLO26x projected-LiDAR pilot used the same 10-frame development and 10-frame holdout comparison sets. At the fixed matched-detection operating point, YOLOv8x / YOLO26x had development depth MAE **1.492 / 1.272 m** (45 / 44 matched cars) and holdout MAE **1.789 / 1.669 m** (41 / 41 matched cars). The matching denominators are shown to avoid treating conditional depth error as full detection accuracy. The sets of matched cars need not be identical, so the lower observed MAE is not a paired improvement claim. These are not official 3D AP results. See `results/yolov8_yolo26_lidar_comparison.csv` and `.png`.

A role-by-role inventory of the tracked result figures, backing numerical files, and limitations is kept in `docs/RESULT_FIGURE_INVENTORY.md`. Presence in Git does not certify visual legibility; inspect final figure exports before choosing dissertation and poster panels.
