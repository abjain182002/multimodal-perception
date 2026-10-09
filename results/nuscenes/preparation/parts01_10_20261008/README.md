# Full nuScenes validation input staging

Staging audit dated 8 October 2026; original evidence imported on 9 October.

The backup-index audit covers all **94,180 required sensor files**, providing complete inputs for **6,019 official validation samples across 150 scenes**, with zero missing required files. Inputs comprise six camera keyframes and LiDAR keyframes with up to nine preceding sweeps. Part 07 keyframe and LiDAR archives were scanned and had zero required-input matches; their zero-match audits are retained.

## Evidence and scope

- `coverage_audit.json`: original coverage and archive provenance record.
- `scene_coverage.json` and `complete_input_sample_tokens.txt`: all 150 scenes and 6,019 unique sample tokens.
- `missing_required_files.txt`: empty after the complete coverage audit.
- `compact_backup_receipts.json`: receipt summaries; full sensor-member maps remain in the supplied source ZIP and Drive backups.
- `part07_zero_match_audits.json`: original records for the two zero-match archives.
- `evidence_manifest.json`: complete source ZIP member hashes and sizes.
- `archive_receipt.json`: import verification and archive identity.

The coverage audit checks backup ZIP indexes, sizes and previously verified receipts. It does not repeat all archive or sensor-content hashes. Import verification checks ZIP CRC, every source-member hash and size, receipt provenance, scene totals and unique sample-token counts. It does not access the large sensor archives.

**This is input preparation, not full-validation accuracy evidence.** Paired inference and global evaluation on all 6,019 samples remain pending. The measured 1,231-sample / 31-scene subset remains a separate historical experiment. Batch inference may be merged by unique sample token; full-split AP/NDS must be computed globally rather than averaged across batches. Earlier parts 1–6 staging records remain unchanged.
