# Ground-truth evaluation replay — 8 October 2026

The same saved predictions were evaluated again against ground truth for the frozen availability-selected 1,231 samples / 31 scenes. Both headline scores reproduced exactly; the largest absolute difference across metric summaries and details was 8.881784197001252e-16, below the 1e-12 tolerance. Evaluation time was excluded from comparison.

| Pipeline | mAP | NDS |
|---|---:|---:|
| LiDAR | 0.6421140063400634 | 0.6920865595499972 |
| Camera–LiDAR | 0.6781470812751105 | 0.7115433802870929 |

## Evidence
- [Paired replay audit](paired_replay_audit.json)
- [Evaluator environment and source hashes](environment_audit.json)
- [Replay script](replay.py) and [completed log](replay.log)
- [LiDAR summary](lidar/metrics_summary.json) and [audit](lidar/replay_audit.json)
- [Camera–LiDAR summary](lidar_camera/metrics_summary.json) and [audit](lidar_camera/replay_audit.json)
- [Complete replay member manifest](backup_receipt.json)
- [Archive verification receipt](archive_receipt.json)

The complete 61-file archive includes both metrics_details.json files and 25 PR/TP/summary PDFs per model. These larger files remain in the verified ZIP on Drive:
`Major_Project/nuscenes_full_validation/bevfusion_subset_replay_verified.zip`.
This compact import preserves the manifest for every source member. Historical original evidence remains in [original_evidence](../original_evidence/).

This is same-evaluator reproduction, not independent cross-implementation verification, full nuScenes validation, a random sample, or a controlled camera ablation. This repository import checked archive integrity and audit consistency; it did not run another ground-truth evaluation.
