# Daily Project Log

This file records work chronologically. `PROJECT_HISTORY.md` remains the synthesized master dissertation/technical record; this log preserves session-level details so that nothing important is lost between larger consolidations.

---

## Falcon + LiDAR single-frame fusion

### Objective
Integrate Falcon Perception with the existing KITTI LiDAR projection and the already frozen median-ROI depth association method on one frame before expanding to a multi-frame experiment.

### Controlled variables
The LiDAR side was intentionally kept unchanged from the YOLO baseline:

- KITTI calibration matrices (`P2`, `R0_rect`, `Tr_velo_to_cam`),
- projected LiDAR points in camera coordinates,
- points associated when their image projection lies inside a 2D detector box,
- minimum of 3 projected LiDAR returns,
- depth estimate = median camera-coordinate LiDAR depth inside the box.

### Changed variable
The camera detector input was changed from YOLOv8x to **Falcon Perception + score-free deduplication at IoU 0.50**.

### Dataset / frame
KITTI training frame:

```text
000149
```

### Environment

```text
~/Major_Project/envs/falcon310
Python 3.10.21
```

### Implementation
Created:

```text
scripts/fuse_falcon_lidar.py
```

The script reuses `adapters/falcon_adapter.py`, converts Falcon detections to project box format, applies the already validated score-free deduplication rule, projects KITTI LiDAR into the camera frame, associates points with each deduplicated box, and estimates median depth.

### Problem encountered
The first runtime attempt failed with:

```text
ModuleNotFoundError: No module named 'matplotlib'
```

### Diagnosis
`matplotlib` had not yet been installed in the dedicated `falcon310` environment. The script itself had already passed `py_compile`, so this was an environment dependency issue rather than a Python syntax problem.

### Fix
Installed Matplotlib into `falcon310`:

```text
matplotlib 3.10.9
```

Import verification succeeded, after which the fusion script ran end-to-end.

### Result
Falcon output on frame `000149`:

```text
Raw Falcon boxes: 11
Deduplicated boxes: 10
Fused detections: 10
```

All 10 deduplicated boxes had at least 3 associated projected LiDAR points.

Median LiDAR depth estimates:

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
| 9 | 9.22 m | 1,629 |

### Qualitative verification
Generated locally:

```text
~/Major_Project/model_outputs/falcon_lidar_fusion/falcon_lidar_000149.png
```

The image was visually inspected. Projected LiDAR returns were aligned with the detected vehicle regions, and the near-to-far ordering of the estimated depths was physically plausible.

A presentation limitation was noted: dense LiDAR points and overlapping labels make the current visualization cluttered. This is a visualization issue, not evidence of a fusion failure, and can be cleaned later for dissertation presentation.

### Git synchronization issue
The first push of the new local commit was rejected because the remote `main` branch already contained newer documentation commits (`PROJECT_HISTORY.md` and the README link).

The repository was synchronized safely with:

```text
git pull --rebase origin main
git push
```

No force-push was used. The rebase succeeded without conflict and the final working tree was clean.

### Final commit

```text
bf3154d — Add Falcon LiDAR single-frame fusion
```

### Dissertation-safe interpretation
This experiment demonstrates successful **single-frame software integration** of Falcon detections with the frozen KITTI projected-LiDAR median-depth association pipeline. It is a qualitative/engineering validation step, not yet a multi-frame accuracy result and not evidence that Falcon+LiDAR outperforms YOLO+LiDAR.

### Next step

- multi-frame Falcon + LiDAR evaluation,
- then controlled YOLO+LiDAR vs Falcon+LiDAR comparison using matched evaluation conditions.

### Progress checkpoint
Working technical-completion estimate after this milestone: **approximately 51%**.

## 2026-09-19 — Controlled YOLOv8x + LiDAR vs Falcon + LiDAR

### Objective
Complete the controlled multimodal comparison while keeping the downstream LiDAR association fixed.

### Environment
- YOLOv8x evaluation: `azitt310`
- Falcon evaluation: `falcon310`
- Apple MPS used for YOLOv8x inference

### Fixed evaluation settings
- same 10-frame development split and 10-frame independent holdout split
- KITTI `Car`
- match IoU = 0.30
- minimum projected LiDAR points = 3
- same calibration/projection
- same V1 median-ROI depth estimator
- YOLOv8x confidence = 0.25
- Falcon deduplication IoU = 0.50

### Verified results
Development:
- YOLOv8x + LiDAR: 45 matched fused cars, MAE 1.492 m, Median AE 0.923 m, RMSE 2.402 m, signed error -0.860 m
- Falcon + LiDAR: 41 matched fused cars, MAE 1.783 m, Median AE 1.105 m, RMSE 2.872 m, signed error -1.261 m

Holdout:
- YOLOv8x + LiDAR: 41 matched fused cars, MAE 1.789 m, Median AE 1.471 m, RMSE 2.355 m, signed error -1.029 m
- Falcon + LiDAR: 42 matched fused cars, MAE 1.655 m, Median AE 1.477 m, RMSE 2.316 m, signed error -0.948 m

### Validation
Both YOLOv8x development and holdout CSVs were independently re-read and the aggregate metrics were reproduced.

### Issues encountered
- `falcon310` did not contain Ultralytics, confirming the need to keep model environments separated.
- the first converted YOLO evaluator failed with `NameError: name 'iou' is not defined`; the shared IoU helper was restored and the full run then completed successfully.

### Interpretation
YOLOv8x was stronger on development, but the ordering did not transfer cleanly to holdout. Falcon produced one extra matched fused car and slightly lower holdout MAE/RMSE, while Median AE was almost identical. Treat this as a split-dependent tradeoff rather than a universal detector ranking.

### Artifacts
- `results/yolov8_falcon_lidar_comparison.csv`
- `results/yolov8_falcon_lidar_comparison.png`
- `scripts/evaluate_yolov8_lidar_multiframe.py`
- `scripts/evaluate_yolov8_lidar_holdout.py`

### Next step
Finish repository documentation and checkpoint this experiment before moving to the next technical stage.

## 2026-09-25 — KITTI CLOCs GT audit and candidate-IoU preparation

- Mac `azitt310`: 84 full-suite tests passed after the GT audit script was pulled.
- Canonical train/validation split: 3,712/3,769 frames, no overlap. Audited all 7,481 labelled KITTI frames; exact Car GT 28,742 before and 28,374 after detector range filtering. Train/validation retained 14,144/14,230 Cars. Saved `results/kitti_clocs_gt_audit.json` at `96c3a37` and confirmed a clean Mac tree.
- CPU reference implementation now computes camera-space rotated 3D IoU for dense candidate matching. PointPillars extraction returns matching camera boxes in candidate order. Seven new synthetic tests pass; locally available suite 43/43.
- Gate: `reference_structural_smoke` has placeholder zero overlaps. Export genuine dense 3D boxes from Colab and independently verify real targets before training or declaring fusion accuracy.

- Added guarded per-frame target builder and GPU export contract. A synthetic end-to-end test exercises split verification, target generation, saved SHA-256 and reload; local subset 49/49 tests passed. Genuine dense candidate export and real-data checks remain the GPU gate.

- Mac confirmation: pulled `e2e3a47`; 97/97 repository tests passed in 0.613 s, with clean `main...origin/main`.
- Tightened CLOCs target input contract: regenerate all rows from saved original 2D detections and projected dense candidate boxes; reject altered 2D geometry, scores, indices or row types. Local subset 52/52 tests passes. Real candidate labels remain pending GPU export.

- Mac confirmation: pulled `6b913a1`; **100/100 tests passed in 0.551 s**, and `main...origin/main` was clean.
- Set daily Phase 1 targets through 10 October in `docs/PHASE1_DAILY_TARGETS_2026-09-25_TO_10-10.md`. Each date has an evidence gate and a GitHub checkpoint or documented blocker. Dissertation chapter drafting starts after 10 October; a continuing GPU outage is escalated rather than recorded as finished work.
