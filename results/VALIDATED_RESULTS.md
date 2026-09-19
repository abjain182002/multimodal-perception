# Experimental Results and Project Notes

Project: **A Software-Only Deep Learning Pipeline for Multi-Modal Object Detection and Robotic Perception Using LiDAR and Camera Fusion Across Multiple Datasets**

I am using this file to keep the main results, settings and observations from my experiments in one place. It gives me a clear record to refer to during supervisor meetings and when I start writing the dissertation. Large datasets, model weights and bulk model outputs are kept outside GitHub because of their size.

## 1. Evaluation setup and notes

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

---

# 6. Main findings so far

1. **The simpler V1 median-ROI LiDAR depth estimator generalized better than the more complicated alternatives tested.**
2. **Projected ROI fusion has identifiable failure modes involving mixed depth layers, truncation, occlusion, crowding, and long-range LiDAR sparsity.**
3. **Raw Falcon has a distinct duplicate/redundancy failure mode on some scenes.**
4. **Transparent score-free deduplication substantially improves Falcon precision/F1 at moderate evaluation IoU thresholds while introducing a small strict-IoU tradeoff.**
5. **YOLOv8x currently has the stronger overall fixed-threshold camera result, while Falcon + dedup has slightly higher recall at some thresholds and offers open-vocabulary flexibility.**
6. **The frozen Falcon+LiDAR median-depth pipeline remained stable on the independent 10-frame holdout and reproduced its saved metrics on an independent CSV integrity check.**

---

# 7. Results I am not claiming yet

The following should **not** be claimed from the current evidence:

- official KITTI AP/mAP performance,
- universal superiority of YOLO over Falcon or Falcon over YOLO,
- a standardized end-to-end runtime ratio,
- completion of the full Azitt PV-RCNN++ pipeline on macOS,
- completed PointPillars / OpenPCDet / BEVFusion results,
- completed nuScenes cross-dataset results,
- completed robustness evaluation,
- completed YOLO+LiDAR vs Falcon+LiDAR controlled multimodal comparison.

---

# 8. Files linked to these results

The main files I am using to keep the work traceable are:

- `PROJECT_HISTORY.md` — detailed chronological technical record, decisions, failures and interpretations.
- `DAILY_LOG.md` — dated working log.
- `results/VALIDATED_RESULTS.md` — this consolidated numerical summary.
- `results/depth_method_comparison.csv` — frozen V1/V2/V3 depth comparison.
- `results/depth_method_comparison.png` — figure for the depth-method comparison.
- `results/camera_detector_comparison.csv` — frozen YOLOv8x vs Falcon camera pilot.
- `results/camera_detector_comparison.png` — figure for the camera comparison.
- `scripts/evaluate_falcon_lidar_multiframe.py` — Falcon+LiDAR development evaluator.
- `scripts/evaluate_falcon_lidar_holdout.py` — Falcon+LiDAR independent holdout evaluator.

Bulk generated outputs and datasets are intentionally excluded from Git because of size and reproducibility policy.

---

# 9. Reproducibility notes

I have kept the following details in the repository:

- exact controlled frame lists,
- thresholds,
- model roles,
- calibration procedure,
- matching rules,
- depth metric definitions,
- development/holdout separation,
- scripts used for the latest Falcon+LiDAR experiment,
- version-controlled frozen result tables and figures for the earlier depth and camera studies.

My next results milestone is the controlled **YOLO+LiDAR vs Falcon+LiDAR** comparison using the same projected-LiDAR and median-depth association rules.


---

# 10. Notes for explaining the work to my supervisor

I am keeping these notes here so I can explain the reasoning behind the experiments clearly in meetings. They are based on the results above and are meant to sit alongside the scripts, CSV files and figures.

## 10.1 What I am trying to do

> My project is looking at how camera detections and LiDAR measurements can be combined in a software-only perception pipeline. I am trying to keep the experiments controlled, so when I change one part of the pipeline I keep the other parts fixed. This makes it easier to see whether a result is coming from the camera detector, the LiDAR association method or the fusion step.

> I am using KITTI as the main dataset at this stage because it gives me the RGB images, Velodyne LiDAR, calibration files and object annotations needed for camera-LiDAR experiments. I plan to use nuScenes later as a second dataset to see how well the approach transfers.

> I am comparing YOLOv8x with Falcon because they represent two different types of camera perception. YOLOv8x is a conventional detector with a fixed class set, while Falcon can work with a natural-language query such as "car". I want to compare them while keeping the LiDAR processing as similar as possible.

## 10.2 Why I kept the median LiDAR depth method

> My first LiDAR depth estimate was simple: after projecting the LiDAR points into the image, I take the points inside a matched 2D detection box and use their median camera-coordinate depth. I used the median because a detection box can also contain some background points, and the median is less affected by a few extreme values than the mean.

> I also tested more complicated methods. The nearest-cluster method was worse, and the edge-aware method looked slightly better on the development data but did not improve MAE or RMSE on the independent holdout. Because of that, I kept the simpler median-ROI method instead of choosing the method that only looked better on the development set.

> This is useful for the project because it shows that I am checking changes on separate data before deciding to keep them.

## 10.3 What I think the negative depth bias means

> The depth estimates are usually slightly smaller than the KITTI ground-truth z value. A likely reason is that many LiDAR returns come from the visible front surface of a vehicle, while KITTI gives the annotated 3D object location in camera coordinates. These are related measurements, but they are not exactly the same physical point on the vehicle.

> I am keeping this bias visible in the results rather than applying a correction without first having a clear reason for doing so.

## 10.4 Why I had to clean Falcon's output

> In the Falcon output I am using, I do not get a normal confidence score for every bounding box. That means I cannot apply standard confidence-ranked NMS in the same way as I would with YOLO without making up a score that the model did not provide.

> Instead, I group strongly overlapping Falcon boxes using IoU and replace each group with the median box coordinates. I use this as a score-free deduplication step.

> Frame 006648 is the clearest example of why this was needed. Falcon returned 101 raw boxes for a scene with 9 KITTI cars. Many of those boxes were repeated around the same vehicles. After the deduplication step, the 101 raw boxes were reduced to 17 representative boxes.

## 10.5 What I found from YOLOv8x and Falcon

> I am not trying to prove that Falcon is better than YOLO or the other way around. The purpose of this comparison is to see what changes when I use a conventional detector versus an open-vocabulary detector under the same evaluation setup.

> On the current holdout, YOLOv8x has better precision and F1, while Falcon with deduplication has slightly better recall at some IoU thresholds. Falcon is also much slower in my current local setup, but it gives me the open-vocabulary behaviour that YOLOv8x does not provide in the same way.

> For me, the useful result is the tradeoff between precision, recall, runtime and flexibility rather than choosing a single "winner".

## 10.6 What I did in the Falcon + LiDAR experiment

> For Falcon + LiDAR, I kept the LiDAR side fixed and changed the camera detector. I use Falcon with deduplication at IoU 0.50, project KITTI LiDAR using the calibration matrices, require at least three projected LiDAR points inside a detection box, and then use the same median-ROI depth estimate that I had already validated earlier.

> On the 10 development frames, I had 41 matched fused cars. The MAE was 1.783 m, the Median AE was 1.105 m, the RMSE was 2.872 m, and the mean signed error was -1.261 m.

> I then ran the same setup on 10 independent holdout frames without changing the method. This gave 42 matched fused cars, with MAE 1.655 m, Median AE 1.477 m, RMSE 2.316 m, and mean signed error -0.948 m.

> The important point for me is that the result stayed in the same range on the holdout instead of getting much worse. The holdout MAE and RMSE were actually lower, although the Median AE was higher. I therefore treat this as evidence that the current simple fusion method is behaving consistently across these two small controlled splits.

## 10.7 Important limitation when I present these numbers

> These are controlled experiments on selected KITTI development and holdout frames. I am not presenting them as official KITTI AP or mAP results.

> The depth errors are also calculated only for fused detections that were successfully matched to KITTI Car annotations at the chosen 2D IoU threshold. Because of that, the MAE and RMSE describe depth quality for matched detections; they are not a complete measure of object-detection performance.

## 10.8 What I consider my contribution at this stage

> My contribution is not just running YOLO or Falcon separately. I have put the camera and LiDAR parts into one controlled pipeline, adapted Falcon output to the KITTI setup, added score-free deduplication for its duplicate boxes, tested different LiDAR depth rules, used separate development and holdout data for decisions, and recorded failure cases instead of only keeping the best examples.

> The next comparison I want to complete is YOLOv8x + LiDAR versus Falcon + LiDAR with the same frames, projection, minimum-point rule, matching threshold and median-ROI depth method. That should make the camera detector the main variable that changes.

## 10.9 Short project update I can give in a meeting

> I started by building a YOLO + projected-LiDAR baseline and checking different ways of estimating depth from the LiDAR points inside a 2D detection. The simpler median method generalized better, so I kept it fixed. I then integrated Falcon, found that it could produce a large number of duplicate boxes, and added a score-free deduplication step because the Falcon output I use does not include normal per-box confidence scores. After validating Falcon on separate holdout frames, I connected it to the same LiDAR depth pipeline. The Falcon + LiDAR result has now been checked on both development and independent holdout frames. My next step is the controlled YOLO+LiDAR versus Falcon+LiDAR comparison, followed by a stronger LiDAR/3D baseline, a stronger fusion experiment, cross-dataset testing and robustness analysis.

## 10.10 How I am keeping the results traceable

I am keeping the exact frame lists, thresholds, scripts, CSV summaries, figures and important Git commits with the results. When I use a number in a meeting or later in the dissertation, I should be able to trace it back to the script and saved experiment that produced it.
