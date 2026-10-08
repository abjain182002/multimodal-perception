# nuScenes input staging — parts 01–06, 8 October 2026

This folder records input availability, not measured detection accuracy.

| Coverage from saved backup indexes | Available | Full validation |
|---|---:|---:|
| Samples with complete required inputs | 3,273 (54.3778%) | 6,019 |
| Complete scenes | 82 | 150 |
| Required sensor files | 51,151 | 94,180 |
| Partial scenes with some complete samples | 0 | — |

All original 1,231 frozen evaluation tokens remain covered. Parts 03–06 keyframes and LiDAR were selectively extracted, checked and backed up; the existing part-03 keyframe backup was reused. File counts include overlaps between keyframe and LiDAR archives; coverage uses their union, not summed counts. Six distribution parts do not imply 60% sample coverage.

## Records
- [Coverage audit](coverage_audit.json) and [scene coverage](scene_coverage.json)
- [Complete-input sample tokens](complete_input_sample_tokens.txt)
- [Compact backup receipts](compact_backup_receipts.json), derived by omitting per-sensor member maps from source receipts
- [Complete evidence manifest](evidence_manifest.json) and [archive verification receipt](archive_receipt.json)

The supplied 1,998,546-byte evidence ZIP passed CRC and all 12 source-member hashes/sizes. The eight backup receipts agree with the coverage audit's recorded archive hashes, sizes and receipt hashes. The complete source ZIP remains on Drive:
`Major_Project/nuscenes_full_validation/nuscenes_parts03_06_staging_verified.zip`.
It includes the available-file list and complete per-sensor receipt maps excluded from this compact import. Sensor backups remain in the same Drive folder.

## Verification scope and next gate

Coverage was computed from backup ZIP indexes, sizes and previously verified receipts. The coverage audit did not rehash full sensor archives or inspect all sensor contents again. This repository import verified the supplied evidence archive and internal consistency; it did not access sensor backups or recompute coverage from full metadata.

Input coverage does not establish representative sampling or detector performance. New inference on added scenes remains pending. The measured [1,231-sample result](../../bevfusion/validation_subset/README.md) and its [ground-truth replay](../../bevfusion/validation_subset/replay_20261008/README.md) retain their original scope.

Before expanded inference, freeze the selected tokens and verify restored sensor inputs. Full-validation reporting requires exactly 6,019 official validation tokens, paired prediction completeness, and one global evaluation; do not average batch mAP/NDS.

[Results overview](../../../README.md)
