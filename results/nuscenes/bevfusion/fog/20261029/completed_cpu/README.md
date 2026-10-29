# Completed synthetic fog evaluation — 29 October 2026

All eight saved model/condition runs were scored successfully with nuScenes SDK 1.2.0 on the frozen 600-frame, 150-scene subset. The original annotations and SDK filters were retained. Both prediction and GT containers were restricted to those 600 frames before metric computation; the 5,419 temporary empty constructor entries were excluded from scoring. All ten classes are supported, with 12,318 filtered annotation instances, including 2,427 pedestrians. These are repeated-frame annotation counts, not unique objects.

## Results

All values below are multiplied by 100; differences are score points.

| Condition | LiDAR mAP | Fusion mAP | Fusion minus LiDAR | LiDAR NDS | Fusion NDS |
|---|---:|---:|---:|---:|---:|
| Fresh baseline | 64.3074 | 68.1406 | 3.8333 | 69.4427 | 71.2940 |
| Fog 1 | 52.5824 | 57.1558 | 4.5734 | 63.2594 | 65.2756 |
| Fog 2 | 43.0942 | 47.2098 | 4.1155 | 58.2904 | 59.8450 |
| Fog 3 | 11.1064 | 12.1456 | 1.0391 | 34.6370 | 35.3177 |

Source: [executed metrics table](fog_metrics.csv), [class metrics](fog_class_metrics.csv), and per-run `metrics_summary.json` files. The [plot](fog_severity_comparison.png) and [PDF](fog_severity_comparison.pdf) show mAP, NDS and pedestrian AP. Use these fresh deterministic baselines for this study rather than substituting historical challenge-study baselines.

## Critical interpretation

Both separately pretrained pipelines deteriorate monotonically in aggregate mAP and NDS across the three tested severity indices. Fusion retains higher aggregate accuracy in each condition, but its advantage is substantially smaller at Fog 3. LiDAR mAP falls by 53.2009 points from its own baseline, while fusion falls by 55.9950 points. Higher absolute fusion accuracy therefore does not imply a smaller loss from baseline at every severity.

Pedestrian AP falls from 86.9301 to 23.4570 for LiDAR and 87.9215 to 24.3039 for fusion. This is a substantial remaining limitation despite fusion's positive aggregate difference. At Fog 3 fusion has lower AP for truck, bus, motorcycle and barrier; motorcycle AP is lower by 4.3848 points. Trailer and construction-vehicle AP are zero for both pipelines. Aggregate comparisons must not conceal class failures.

Mean translation error is slightly lower for LiDAR at Fog 1/2 than at baseline while its mAP decreases. This does not establish improved overall localisation: the TP error metric is conditional on matched detections and changing detection coverage. Increasing predicted-box counts likewise do not establish more correct detections.

The protocol provides a paired synthetic input challenge, not real-fog validation. Camera fog is heuristic and not depth calibrated; the camera and LiDAR severities do not encode a shared physical visibility distance. LiDAR source noise assumptions and retained outliers are additional limitations. There is one seeded corruption realisation, correlated frames from 150 scenes, no confidence intervals, and no significance or deployment-safety claim. Independent seeds at each severity also mean this is not an isolated deterministic severity-dose experiment. Separately pretrained checkpoints prevent attributing the performance difference solely to adding cameras. No snow, wind, glare or natural-disaster performance follows from these results.

## Verification and preservation

The complete uploaded CPU ZIP is preserved here: `fog_cpu_evaluation_20261029T160939790622Z.zip` (1,275,838 bytes; SHA256 `f799b3272e6c914c66aae0332984e7bbd73fc89e74b546a0620511737b3636ea`). All 53 payloads, exact member set and CRC passed local verification. It contains all metric details, logs, source, configuration and audit files; this directory exposes selected key outputs for reading.

The [local evidence review](verification.json) confirms the common GT fingerprint, eight prediction hashes linked to the accepted inference summary, deployed worker/launcher identity, all 19 reused metadata hashes against the prior accepted reference, class/AP/mAP/NDS arithmetic and CSV consistency, and visual plot review. This is review of executed Colab evidence; no local re-evaluation or raw-sensor rehash was performed. The eight full prediction archives and per-condition metric capsules are recorded on Drive in the embedded receipts; those separate metric ZIPs were not attached for independent local opening.

## Remaining work

The bounded fog experiment, CPU scoring and evidence review are complete. Integrate these findings with the completed natural rain/night group analysis and earlier controlled challenges in the dissertation; retain snow, wind, glare and disasters as literature-supported limitations/future validation. Complete report-wide references, presentation and poster. No additional GPU experiment is required to close this fog study; grades depend on the full assessed submission.
