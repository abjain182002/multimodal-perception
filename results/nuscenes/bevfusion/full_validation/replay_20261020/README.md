# Full-validation ground-truth replay — 20 October 2026

Both preserved full-validation prediction files were evaluated again on CPU against official nuScenes validation ground truth: **6,019 samples / 150 scenes per model**. Every compared summary/detail value reproduces within absolute tolerance **1e-12**; maximum absolute difference **8.881784197001252e-16** for each model. Evaluation time is excluded from comparison. This closes the previously recorded fresh full-validation replay gap.

| Pipeline | mAP | NDS | Numeric values compared | Differences above tolerance |
|---|---:|---:|---:|---:|
| LiDAR-only | 0.6443012674636525 | 0.6923533233234365 | 32,451 | 0 |
| Camera–LiDAR | 0.6830990666994801 | 0.7113516737651924 | 32,451 | 0 |

These are same-evaluator reproduction results. They do not independently cross-verify evaluator implementation, reproduce upstream training or explain the checkpoint-linked reference discrepancy. No new inference was performed. The original full-validation metrics are unchanged.

## Preserved execution evidence

- [Paired replay audit](original_evidence/replay/paired_replay_audit.json).
- [Executed replay script](original_evidence/replay/replay_full.py), [LiDAR log](original_evidence/replay/lidar_replay.log) and [fusion log](original_evidence/replay/lidar_camera_replay.log).
- [LiDAR audit](original_evidence/replay/lidar/replay_audit.json), [summary](original_evidence/replay/lidar/metrics_summary.json) and [detailed metrics](original_evidence/replay/lidar/metrics_details.json).
- [Fusion audit](original_evidence/replay/lidar_camera/replay_audit.json), [summary](original_evidence/replay/lidar_camera/metrics_summary.json) and [detailed metrics](original_evidence/replay/lidar_camera/metrics_details.json).
- [Evaluator environment](original_evidence/setup/environment_audit.json), [setup log](original_evidence/setup/cpu_setup.log), [metadata restoration audit](original_evidence/setup/metadata_restoration_audit.json) and per-model detection configurations.
- Original reference summaries, detailed metrics and evaluation audits under `original_evidence/original_reference/evaluation/`.
- [Drive backup receipt](original_evidence/replay/backup_receipt.json), [source member manifest](original_evidence/evidence_manifest.json) and [archive import receipt](archive_receipt.json).

Python 3.10.22, nuScenes-devkit 1.2.0, NumPy 1.26.4, SciPy 1.15.3, Matplotlib 3.8.4, scikit-learn 1.7.2 and Shapely 2.0.7 match the recorded versions. All four recorded evaluator source hashes match. The detector inference hardware is unrelated to the CPU replay, which consumes saved detections.

Each model ran in a fresh process. Checks cover input prediction hashes; restored metadata sizes/hashes; original metric hashes; detection configuration equality; official scene/sample tokens; filtered per-class GT counts; and recursive comparison of both metric JSON files, excluding `eval_time`. The replay checks 121,861 filtered GT boxes. Invalid/undefined class-specific metric values are compared with matching NaN handling.

## Reproduction

The executed script retains its Colab paths and original argument interface. Restore the same metadata, full prediction files and verified 10 October evidence archive to the paths recorded in the script. Recreate the pinned CPU environment and check source hashes before running. From the prepared Colab environment, execute one process at a time:

```bash
/content/camera_challenges_cpu310/bin/python -u replay_full.py lidar /content/full_validation_replay_20261020/new_run
/content/camera_challenges_cpu310/bin/python -u replay_full.py lidar_camera /content/full_validation_replay_20261020/new_run
```

Use a fresh output directory. The setup log records package installation and source checks; the script asserts the environment, reference files, metadata and predictions before evaluation. Licensed metadata and bulk predictions remain outside Git. Curves were not regenerated because their underlying numerical metric files are compared directly and the original curves are already preserved.

## Archive and import verification

Source archive `full_validation_replay_evidence_20261020.zip`: **833,241 bytes**, SHA-256 `95eb6f99c1a495533bf62d1358fe7edc8527b7065ad032a7e646c2ad354b5a7a`. All **23 members** are imported unchanged. The 22 manifest-listed files match their saved sizes and SHA-256 values; the remaining member is the manifest itself. The supplied Colab output reports a hash-verified Drive copy.

The repository import additionally compared replayed summary/detail JSON values with the uploaded originals and obtained the same numeric counts and maximum errors. Original reference summary/detail bytes were checked against the previously verified 10 October full-validation archive. The import checks archive integrity and consistency of execution evidence; it did not run another ground-truth evaluation locally.

Drive execution folder: `Major_Project/nuscenes_full_validation/full_validation_replay_20261020/runs/20261020T103156Z`.

The [upstream comparison note](../../../../benchmarks/BENCHMARK_COMPARISONS.md) still records an unexplained reproduction discrepancy. Successful replay establishes that this project's original metrics can be recomputed from the preserved predictions under the same evaluator; it does not identify a cause for the upstream difference.
