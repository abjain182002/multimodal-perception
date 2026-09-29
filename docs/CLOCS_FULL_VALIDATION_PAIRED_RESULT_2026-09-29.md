# Full KITTI CLOCs paired validation diagnostic (29 September 2026)

## Scope and provenance

The frozen YOLO26x and 128-frame ratio-0 CLOCs checkpoint were run with pinned PointPillars on all **3,769 ordered KITTI SECOND validation frames**. Inference saved one JSON per frame; the evidence ZIP has SHA-256 `1b034d06f49c343dfc4a21c80339481e37db455f237a19f3db3764d3a1306413`. The Mac download matched this digest. The repository recorder verified all 3,769 files, ordered prediction digest, audit references, and manifest before generating figures. Full per-frame predictions remain outside Git.

The two methods were scored on the same frames, annotations, pinned KITTI evaluation code and Car strict 0.70 IoU AP40 protocol. PointPillars generated 24,180 retained detections and CLOCs rescoring generated 19,036. The four-frame pilot was separate engineering evidence; it did not select parameters on this validation set.

## Same-run measured result

All values are AP40 points (percent).

| Metric | Difficulty | PointPillars in this run | CLOCs in this run | CLOCs minus PointPillars |
|---|---|---:|---:|---:|
| 2D | Easy | 88.0113 | 96.3150 | +8.3037 |
| 2D | Moderate | 85.8409 | 93.2839 | +7.4429 |
| 2D | Hard | 81.7690 | 87.8788 | +6.1098 |
| BEV | Easy | 86.3592 | 92.0486 | +5.6895 |
| BEV | Moderate | 82.8221 | 88.5903 | +5.7682 |
| BEV | Hard | 78.7595 | 83.0762 | +4.3167 |
| 3D | Easy | 82.5604 | 89.8204 | +7.2600 |
| 3D | Moderate | 74.2256 | 79.1601 | +4.9345 |
| 3D | Hard | 70.0869 | 73.5984 | +3.5115 |

Source: `results/clocs_full_val_paired/paired_diagnostic.json` (pending local commit). The exact nine values and the parity failure are independently copied from the hash-verified evidence ZIP by `scripts/record_clocs_full_val_paired.py`.

## Historical baseline discrepancy

The earlier PointPillars validation experiment reported **79.1299** strict 3D Moderate AP40, 23,844 retained predictions and a different full prediction SHA-256. This run measured **74.2256** and 24,180 predictions. The largest difference among the nine historical AP40 values is **8.0934 points** (2D Easy). The full evaluator correctly set `BASELINE_PARITY_PENDING` and withheld a fusion result as a reproduced-baseline comparison.

A six-frame native MMDetection3D check found identical retained counts and scores, matching dimensions and yaw, and 2D-coordinate differences no larger than 0.000061 pixels. That rules out a general conversion failure on those six frames; it does not establish parity across 3,769 frames. The original `kitti_infos_val.pkl` and `pred_instances_3d.pkl` were not found under the Mac project root, so the cause of the historical difference remains unresolved.

The observed **79.1601** fusion 3D Moderate AP40 is compared only with **74.2256 from the same run**. Its numerical proximity to the earlier **79.1299** baseline is not evidence of a cross-run gain. This is a paired full-validation diagnostic, not a reproduced historical baseline or an official KITTI test-server submission. No tuning should be performed on this validation result.

## Next checks

1. Preserve the baseline parity failure and paired result together.
2. Compare native PointPillars conversion and KITTI annotations on a broader, preselected set of frames; recover the original baseline artifacts if available.
3. Analyze error counts by range, difficulty and occlusion from the per-frame ZIP using a frozen matching definition. Report paired uncertainty or scene variability without tuning the model on validation.
4. Include this limitation in the dissertation and poster if parity remains unresolved.
