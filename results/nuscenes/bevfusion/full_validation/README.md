# BEVFusion — full nuScenes validation

Evaluated 10 October 2026: **6,019 samples / 150 official validation scenes**, two official pretrained pipelines.

| Pipeline | mAP × 100 | NDS × 100 |
|---|---:|---:|
| LiDAR-only | 64.4301 | 69.2353 |
| Camera–LiDAR | 68.3099 | 71.1352 |
| Difference (points) | +3.8798 | +1.8998 |

## Find the evidence

- [Findings and limitations](FINDINGS.md)
- **Figures:** [overall comparison](figures/comparison.png), [class AP](figures/class_ap.png), [TP errors](figures/tp_errors.png), [class support](figures/class_support.png). Each has PNG, PDF and SVG versions.
- **Tables:** [headline/error metrics](tables/comparison.csv), [class AP and support](tables/class_ap.csv).
- **Original evaluation:** [paired report](original_evidence/evaluation/paired_evaluation.json), [LiDAR metrics](original_evidence/evaluation/lidar/metrics_summary.json), [camera–LiDAR metrics](original_evidence/evaluation/lidar_camera/metrics_summary.json). Original per-class PR/TP PDFs are in each model's `plots/` folder.
- **Protocols and source:** `original_evidence/inference/batch_01/` through `batch_03/`; `original_evidence/preparation/`; [evaluator script](original_evidence/evaluation/evaluate_full.py); [environment](original_evidence/evaluation/environment_audit.json).
- **Integrity:** [archive receipt](archive_receipt.json), [111-member original manifest](original_evidence/evidence_manifest.json), [figure hashes](figure_manifest.json).
- **Bulk prediction locations:** [recorded Drive paths and hashes](original_evidence/raw_prediction_locations.json). Raw sensor data, checkpoints and bulk predictions are outside Git.

Recreate figures from the repository root with `python scripts/plot_bevfusion_full_results.py`.

This is full validation, not the historical availability-selected subset. It compares pretrained pipelines, not a controlled camera ablation. A fresh full ground-truth CPU replay completed on 20 October, reproducing both models' summary/detail values within 1e-12; [replay evidence and execution records](replay_20261020/README.md). This is same-evaluator reproduction. Independent cross-implementation verification remains unperformed.

The 9.35 MB sensor-filename batch plan remains in the verified source ZIP; the committed manifest records its hash. All three ordered sample-token lists and the compact merge audit are committed.

