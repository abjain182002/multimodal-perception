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

Source: `results/clocs_full_val_paired/paired_diagnostic.json` (commit `fbe72a3`). The exact nine values and the parity failure are independently copied from the hash-verified evidence ZIP by `scripts/record_clocs_full_val_paired.py`.

## Historical baseline discrepancy

The earlier PointPillars validation experiment reported **79.1299** strict 3D Moderate AP40, 23,844 retained predictions and a different full prediction SHA-256. This run measured **74.2256** and 24,180 predictions. The largest difference among the nine historical AP40 values is **8.0934 points** (2D Easy). The full evaluator correctly set `BASELINE_PARITY_PENDING` and withheld a fusion result as a reproduced-baseline comparison.

A six-frame native MMDetection3D check found identical retained counts and scores, matching dimensions and yaw, and 2D-coordinate differences no larger than 0.000061 pixels. That rules out a general conversion failure on those six frames; it does not establish parity across 3,769 frames. The original `pred_instances_3d.pkl` was recovered from the earlier Google Drive evaluation folder and verified against the frozen SHA-256 `2905f9c15fd04a6055dca99753b6fb6e6585492371608c5a625e3cc6ded764d3`. An ordered 3,769-frame comparison found 2,393 frames with differing detection counts, 1,278 with equal counts but differing fields, and 98 matching under the audit's numeric tolerances; the historical and current totals were 23,844 and 24,180. The 98 matches may include empty detections. A replay of the historical pickle against this run's saved ground truth using the same pinned evaluator reproduced all nine historical AP40 values exactly (maximum absolute difference 0.0 points, including 3D Moderate 79.12987417920098). The measured baseline discrepancy is therefore due to differing prediction artifacts; the precise inference or postprocessing cause remains unresolved. Source: `results/clocs_full_val_paired/historical_prediction_comparison.json`, `scripts/replay_clocs_historical_pointpillars.py`, and the replay output from Kaggle (JSON artifact pending transfer).

The observed **79.1601** fusion 3D Moderate AP40 is compared only with **74.2256 from the same run**. Its numerical proximity to the earlier **79.1299** baseline is not evidence of a cross-run gain. This is a paired full-validation diagnostic, not a reproduced historical baseline or an official KITTI test-server submission. No tuning should be performed on this validation result.

## Best-IoU coverage diagnostic

A separate CPU geometry pass over the same 3,769 saved frames examined **14,385 labelled Cars** (including cases that KITTI AP may ignore). It counted a Car as covered if any retained prediction had rotated camera-frame 3D IoU ≥ 0.70. This is not one-to-one matching, precision, recall, or official AP.

| GT depth | Labelled Cars | PointPillars covered | CLOCs covered | PointPillars coverage | CLOCs coverage |
|---|---:|---:|---:|---:|---:|
| <20 m | 5,807 | 5,369 | 5,279 | 92.4574% | 90.9075% |
| 20–40 m | 5,632 | 4,220 | 4,255 | 74.9290% | 75.5504% |
| 40–60 m | 2,404 | 1,015 | 1,061 | 42.2213% | 44.1348% |
| 60+ m | 542 | 48 | 52 | 8.8561% | 9.5941% |
| **All** | **14,385** | **10,652** | **10,647** | **74.0494%** | **74.0146%** |

CLOCs rescued **317** labelled Cars missed by PointPillars and lost **322** that PointPillars covered. There were **288** frames with at least one rescue and **279** with at least one loss. At <20 m, losses exceeded rescues (152 versus 62); at 40–60 m, rescues exceeded losses (74 versus 28). Frame `001583` gained three covered Cars, while frame `000393` lost three. These are diagnostic examples requiring visual inspection before causal explanation.

The near-zero change in coverage and positive same-run AP40 change are compatible with changes in detection ranking or false positive burden; this is an inference from the two audits, not a measured decomposition of AP. Source: `results/clocs_full_val_paired/coverage_diagnostic.json`, `coverage_frames.csv`, and the corresponding range figure, generated by `scripts/analyze_clocs_full_val_coverage.py` (local results pending commit).

## Selected qualitative cases

The evidence ZIP generated paired image overlays and a 3D-IoU audit for three preselected diagnostic scenes:

- **001583 (three rescues, zero losses):** one Car at 42.53 m changed from best retained IoU 0 to 0.8212. Two others at 6.07 m and 15.48 m crossed the 0.70 threshold. The 6.07 m change is small (0.6929 to 0.7286), so it should be described as threshold-sensitive.
- **000393 (zero rescues, three losses):** two Cars at 25.26 m and 32.13 m changed from best retained IoU 0.7893 and 0.7718 to zero; another at 13.38 m fell from 0.7053 to 0.6868. This illustrates a real failure of the frozen rescoring rule on this scene.
- **000831 (two rescues, one loss):** two near Cars at 10.83 m and 12.91 m gained IoU 0.8093 and 0.8051 from zero, while one at 28.07 m lost its 0.7763 match.

Sources: `results/clocs_full_val_paired/cases/case_studies.json` and the paired PNG/PDF overlays (local outputs pending commit). Cyan boxes indicate labelled Cars; orange boxes are 2D projections of retained 3D detections; green/red highlight GT whose best 3D IoU crossed the threshold. These are illustrative cases, not one-to-one KITTI AP matches or proof that a particular camera detection caused each change.

## Next checks

1. Preserve the coverage diagnostic with the paired AP and baseline parity failure.
2. Trace the changed predictions through the historical and current inference settings, input preprocessing, model versions, and postprocessing. Preserve the replay JSON artifact and its hashes.
3. Analyze error counts by range, difficulty and occlusion from the per-frame ZIP using a frozen matching definition. Report paired uncertainty or scene variability without tuning the model on validation.
4. Include this limitation in the dissertation and poster if parity remains unresolved.
