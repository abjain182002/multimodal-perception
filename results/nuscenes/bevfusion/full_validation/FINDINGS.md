# Full-validation findings

> **Analysis status update, 15 October 2026:** Six annotated case studies and controlled camera-availability, image-darkening and LiDAR point-reduction studies are now preserved. A separate paired T4 runtime protocol is complete. These later studies do not change the original full-validation metrics or turn inference-loop wall time into model latency. See the [results index](../../../README.md) for links. A fresh full-split replay subsequently completed on 20 October; see the update below.

## Result and scope

Official pretrained camera–LiDAR BEVFusion achieved mAP 0.6830990666994801 and NDS 0.7113516737651924; LiDAR-only achieved 0.6443012674636525 and 0.6923533233234365. Differences are **+3.8798 mAP points and +1.8998 NDS points**.

All 6,019 samples from all 150 official validation scenes were included. Storage batches of 2,368, 2,368 and 1,283 samples were merged before one global evaluation per model; batch metric scores were not averaged. The exact official prediction/ground-truth token match, duplicate-free merge and configuration consistency across batches passed.

## Per-class detection and support

| Class | LiDAR AP × 100 | Camera–LiDAR AP × 100 | Difference (points) | Filtered GT boxes |
|---|---:|---:|---:|---:|
| car | 86.97 | 89.36 | +2.39 | 53043 |
| truck | 59.61 | 63.97 | +4.36 | 9260 |
| bus | 73.05 | 76.37 | +3.32 | 2062 |
| trailer | 44.60 | 47.71 | +3.11 | 2331 |
| construction_vehicle | 25.47 | 28.85 | +3.39 | 1513 |
| pedestrian | 86.37 | 87.87 | +1.49 | 23500 |
| motorcycle | 69.55 | 75.65 | +6.11 | 1855 |
| bicycle | 54.87 | 62.52 | +7.64 | 1824 |
| traffic_cone | 74.03 | 79.13 | +5.10 | 10570 |
| barrier | 69.78 | 71.67 | +1.90 | 15903 |

Camera–LiDAR AP is higher for all ten classes. The largest differences are bicycle (+7.64 points), motorcycle (+6.11) and traffic cone (+5.10). Class support varies strongly, and construction-vehicle AP remains low in both pipelines; the mechanism is not established by aggregate metrics.

## Error tradeoffs

| Metric (lower is better) | LiDAR | Camera–LiDAR |
|---|---:|---:|
| mATE (m) | 0.283721 | 0.278899 |
| mASE | 0.252398 | 0.254753 |
| mAOE (rad) | 0.294990 | 0.302263 |
| mAVE (m/s) | 0.280292 | 0.279405 |
| mAAE | 0.186572 | 0.186659 |

Translation and velocity errors decrease; scale, orientation and attribute errors increase. Higher overall mAP/NDS does not establish improvement in every localization or semantic measure.

## Provenance and verification

- MMDetection3D commit: `fe25f7a51d36e3702f961e198894580d83c4387b`; Torch 2.1.2+cu121; Tesla T4; seed 20261002.
- Both official checkpoint hashes, strict checkpoint loading in the executed inference source, ordered sample tokens, finite predictions and CPU formatting are recorded.
- nuScenes-devkit 1.2.0, detection_cvpr_2019 configuration and four evaluator source hashes match the earlier subset evaluation; eval_set is the official `val`.
- The supplied 2,118,593-byte ZIP SHA-256 is `37a5882333283e882d21d8f165b308f79a5039a8e1064334ad480b93bd665269`. All 111 original members match their manifest byte sizes and SHA-256 values. Original evidence bytes are preserved.
- Stored mAP equals the mean of class APs, and stored NDS equals `(5*mAP + sum(tp_scores))/10`, within 1e-12. This arithmetic check is not another ground-truth evaluation.
- Raw merged prediction hashes and private Drive paths are recorded; the raw predictions are not in this compact ZIP or Git. The uploaded execution log reports verified Drive prediction hashes and output backups.

## Limitations and interpretation

- This compares two pretrained pipelines with different learned components and training histories, not a controlled camera ablation, training by this project, or a new architecture.
- Official validation has been inspected iteratively; it is not an untouched final test split. Statistical significance and robustness under corruption have not been tested.
- This full-validation result removes the earlier subset coverage restriction. Mini/subset/full results overlap and are not independent replicates. Full validation does not demonstrate performance on other domains or a deployed robot.
- Inference-loop wall time includes I/O, backup and formatting; it is not a synchronized model latency benchmark.
- The original full evaluation and a fresh same-evaluator full-split CPU replay have completed. All compared summary/detail values reproduce within 1e-12, excluding evaluation time; [20 October replay](replay_20261020/README.md). Independent cross-implementation verification remains unperformed.
- Six selected nuScenes visual case studies were preserved on 13 October, including successes and failures. Selected cases do not establish population-level robustness. Keep sensor backups while planned robustness work remains.

## Figure captions

- Overall comparison: full-validation mAP and NDS, displayed ×100, for two official pretrained pipelines; descriptive values without confidence intervals.
- Class AP: official class AP ×100 and point differences for identical validation samples.
- TP errors: native metric units on separate axes; lower is better. L denotes LiDAR, L+C camera–LiDAR.
- Class support: ground-truth box counts after the official distance, sensor-point and bike-rack filters.




## Upstream-reference audit — 20 October 2026

The [bounded reference audit](../reference_audit_20261020/README.md) confirms full published checkpoint hashes and equality of all six saved model dictionaries/test pipelines with the pinned reference. The linked validation logs report LiDAR mAP/NDS ×100 64.85/69.56 (epoch 20) and fusion 68.58/71.39 (epoch 5), consistent with the rounded model-table context. Project differences against those printed values are approximately −0.42/−0.32 and −0.27/−0.25 points. The logs record historical runtime/configuration differences, but none has been isolated as the cause. Upstream SDK-version/annotation equivalence and checkpoint metadata-to-log-row identity remain unverified. The discrepancy remains unexplained; the project's full-validation replay and within-project paired comparison remain valid at their stated scope.

