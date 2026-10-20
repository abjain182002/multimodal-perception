# CLOCs saved-record audit — 20 October 2026

**Outcome: bounded audit complete; cause of seed-20260929's Easy-AP40 decline remains inconclusive.** The score-blending branch is retained as an unfinished exploratory pilot, with no full-AP40 gain claimed. This replaces the earlier status of an unperformed existing-record audit; it does not claim a resolved failure mechanism.

## Evidence examined

The [audit script](../../../../scripts/audit_clocs_saved_seeds.py) checks 26 saved JSON records plus the current repository trainer, listed with hashes in [source_manifest.json](source_manifest.json). Inputs match the GitHub snapshot at `a6ab95b2e116df94c427d7b8c091aedcb5f07e12`. Local stale plotting manifests were excluded. The audit inspects recorded settings, input hashes, splits, learning histories, checkpoint-selection/reload reports, baseline parity and saved AP40. It does not rehash bulk examples or checkpoints, attribute errors to individual validation objects, or recompute AP40.

### Checks that passed

- The three recorded protocols are identical after removing only training seed and output folder. All record the same source hashes, environment, learning rate 0.002, 12 epochs and negative ratio 0.
- Input audits, the 96 optimization frame IDs, 32 development frame IDs and split seed 20260928 agree across runs. Optimization and development sets are disjoint.
- All histories contain epochs 1–12 with identical selected positive/negative counts and development candidate support. Each selected checkpoint is the maximum recorded development candidate AP, at epoch 12; the recorded checkpoint reload candidate-AP error is zero.
- All three AP40 records cover 3,769 reduced-FOV validation frames, record the same validation input-manifest hash, and reproduce all nine frozen PointPillars baseline metrics exactly. This is a check of saved records; raw input-manifest contents were not rehashed today.
- The development-selected seed is 20260928 and its recorded checkpoint hash agrees with the selection file. That file states validation results were not used for the original selection. Later validation inspection was iterative.

## Quantitative findings

| Seed suffix | Best dev candidate AP ×100 | First epoch above dev candidate baseline | Final fusion detections | Easy 3D AP40 | Moderate 3D AP40 | Hard 3D AP40 |
|---|---:|---:|---:|---:|---:|---:|
| 928 | 87.7935 | 4 | 18,686 | 90.7684 | 79.8616 | 74.8719 |
| 929 | 86.6960 | 6 | 18,039 | 82.5581 | 76.2666 | 73.1530 |
| 930 | 86.6751 | 3 | 18,119 | 89.8051 | 78.7616 | 73.4648 |

Table: fixed train/development split; epoch selected by development candidate AP. Detection AP40 is KITTI Car 3D, strict IoU 0.70, 3,769 validation frames. The two AP measures describe different objects and evaluation procedures. Final detection counts cover all outputs, not Easy-only true positives. Full precision and all nine AP40 measures are in [seed_audit_table.csv](seed_audit_table.csv); per-epoch values are in [epoch_audit_table.csv](epoch_audit_table.csv).

Seed-29's Easy decline relative to PointPillars is **7.7869 AP points** (82.5581 versus 90.3449). Relative to seed-30, its development candidate AP is **0.0209 points higher**, yet its Easy 3D AP40 is **7.2471 points lower**. The 80-detection difference in total outputs does not identify which candidates changed or explain this discrepancy. Seed-29 also has lower 2D and BEV AP than the other two seeds, so the result is not confined to the 3D measure.

Seed-29 exceeds the frozen development candidate baseline only at epoch 6, versus epochs 4 and 3 for seeds 28 and 30. All three histories continue improving candidate AP through epoch 12. These observations describe different learning trajectories; they do not prove undertraining, overfitting, miscalibration, a local optimum or a faulty checkpoint. Similar final candidate AP does not establish similar difficulty-specific detection AP40 after post-processing.

## Reproduction source gap

All run protocols record executed trainer SHA-256 `23e798e098a639f70e582093a20a93e56bb2b7a8f1db7033c2de96f6a60f54db`. The current repository `scripts/train_clocs_128.py` has SHA-256 `7bb2cc59c3ea43448567bd74df9e102611fe7dee71736215fff780fc52a80bc6`. They are not the same source bytes. In particular, the recorded aligned runs expose a fixed `split_seed` separately from training seed; the current generic trainer uses its seed for its split. The original run records establish equal splits, but the generic script alone is insufficient to reproduce that executed configuration.

Recover the exact hash-matching Kaggle trainer before rerunning the study. This provenance gap does not explain why seed-29 differs: all three run records name the same executed trainer hash. No replacement source has been reconstructed and presented as original.

## Score-blend pilot: explicit scope decision

The earlier 16-frame pilot used 12 optimization and four held-out frames. A grid searched on the four held-out frames selected logit-prior coefficient 1.5; that maximum is exploratory because selection and scoring reuse those frames. A separate saved analysis selected coefficient **1.875 on the 12 optimization frames** and reported four-frame candidate AP **0.8432663566**, versus PointPillars **0.7965761635**: a difference of **4.6690 candidate-AP points** on 427,994 candidates with 508 positives. This is not KITTI detection AP40 or evidence of calibrated probabilities.

All 16 pilot frame IDs subsequently appear in the 96-frame optimization set of the aligned study. The pilot is therefore neither an independent replication of the later seed study nor a validated remedy for seed-29. Full AP40 testing of the blend remains unperformed. The dissertation should preserve it as an unfinished exploratory branch, report the selection scope, and exclude it from headline detector comparisons. No new blend coefficient is selected from the iteratively inspected full validation set.

## What would resolve the causal question

Further attribution requires the exact executed source, checkpoint-to-prediction provenance for each seed, per-frame pre/post-processing candidates and scores, selected candidate identities and difficulty-specific GT assignments. These would distinguish ranking, thresholding and suppression effects from localisation or class errors. Aggregate detection totals and AP40 alone cannot do so. The audit neither excludes every implementation issue nor claims that additional training would solve the problem.

## Dissertation discussion draft

The small-data CLOCs implementation produced inconsistent detection gains across three training seeds despite a common recorded input set and evaluation protocol. The saved-record audit found no difference in the recorded train/development split, source hashes or hyperparameters beyond the training seed, and all paired baseline scores reproduced exactly. However, development candidate AP did not reliably predict difficulty-specific detection AP40: seed-29 slightly exceeded seed-30 on the development ranking metric while scoring substantially lower on Easy 3D AP40. This is consistent with a limitation of the selection proxy in this study, but the aggregate records do not identify a causal mechanism. The result therefore supports a bounded negative finding about this implementation and protocol rather than a general conclusion that camera–LiDAR fusion fails.

The score-combination pilot illustrates a possible follow-up direction, but its limited candidate-level evidence cannot establish a detector improvement or explain the seed effect. A larger, independently evaluated training study and object-level post-processing diagnosis would be required before claiming a remedy. For the present dissertation, reporting all three seeds and the unresolved source/provenance limits provides a more defensible account than selecting only the favourable run. The completed nuScenes case diagnostics address score eligibility in another pipeline; they are not transferred as an explanation for CLOCs.

## Reproduce this audit

From the repository root:

```bash
python scripts/audit_clocs_saved_seeds.py --repo . --output /tmp/clocs_saved_record_audit
```

The standard-library script regenerates both CSVs, `audit.json` and the source manifest from the saved records. Its status explicitly retains an inconclusive causal outcome. No GPU is required. A future source recovery changes the current-trainer comparison and should be recorded as a new audit, while preserving this dated snapshot.
