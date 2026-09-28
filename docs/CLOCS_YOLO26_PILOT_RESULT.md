# YOLO26x CLOCs pilot result

Date: 28 September 2026

This milestone trains the CLOCs fusion network using frozen PointPillars LiDAR candidates and frozen YOLO26x camera associations from 16 selected KITTI training frames. The 12-frame optimization subset and four-frame holdout subset are fixed in the pilot report.

## Leakage-controlled result

Calibration is selected on the 12 optimization frames only. A logit-prior score blend is then evaluated once on the four held-out frames.

| Measure | Value |
|---|---:|
| PointPillars holdout AP | 0.796576 |
| Selected calibration parameter | beta = 1.875 |
| Calibrated CLOCs holdout AP | 0.843266 |
| Absolute improvement | +0.046690 AP |
| Holdout candidates | 427,994 |
| Holdout positive candidates | 508 |

The four holdout frame IDs are 000149, 000972, 002496 and 004838. The optimization frame IDs and full parameter grid are recorded in results/clocs_yolo26_pilot_calibration_split.json.

## Interpretation

The CLOCs network learns a useful association signal, but its standalone score is weaker than the frozen PointPillars score on this pilot. Retaining PointPillars as a prior and adding the learned CLOCs logit improves the held-out ranking by 4.669 percentage points in this controlled pilot.

This is not a full KITTI validation AP and does not establish generalisation to the complete 3,769-frame validation split. The calibration parameter was selected without using the four holdout labels. The result must be extended to a larger, independently held-out export before it is described as the final fusion performance.

## Provenance

- Training report: results/clocs_yolo26_pilot_report.json
- Learning curve: results/clocs_yolo26_pilot_learning_curve.csv
- Leakage-controlled audit: results/clocs_yolo26_pilot_calibration_split.json
- Figure: results/clocs_yolo26_pilot_calibration_split.png
- Pilot trainer: scripts/train_clocs_pilot.py
- Calibration audit: scripts/audit_clocs_pilot_calibration_split.py
- Frozen camera identity: YOLO26x, Ultralytics 8.4.132, project weight SHA-256 recorded in the training report
- Frozen LiDAR detector: PointPillars Car KITTI checkpoint and configuration hashes recorded in the input package