# Multimodal Object Detection and Robotic Perception

A software-only MSc project evaluating camera, LiDAR and fusion pipelines on KITTI and nuScenes. The aim is to examine detection accuracy, depth estimation, failure cases and computational cost relevant to robotic perception. Robot navigation or hardware deployment has not been demonstrated.

## Start here

| What you need | Where to go |
|---|---|
| Benchmark results and experiment folders | [Results overview](results/README.md) |
| Figures and selected case studies | [Figure gallery](docs/FIGURE_GALLERY.md) |
| Measured results and limitations | [Validated results](results/VALIDATED_RESULTS.md) |
| Technical setup and detailed project status | [Project guide](docs/PROJECT_GUIDE.md) |
| Code entry points | [Scripts guide](scripts/README.md) |
| Environment versions and external sources | [Requirements](requirements/ENVIRONMENTS.md), [external repositories](requirements/EXTERNAL_REPOSITORIES.md) |
| Decisions and chronological records | [Project history](PROJECT_HISTORY.md), [daily log](DAILY_LOG.md) |

## Main measured comparisons

Each row is a separate evaluation; KITTI AP40 and nuScenes metrics are not interchangeable.

| Experiment | Evaluation scope | Result | Evidence |
|---|---|---|---|
| PointPillars vs PV-RCNN | KITTI Car, 3,769 validation frames, strict IoU 0.70 | Moderate 3D AP40: 79.13 vs 84.36 | [PV-RCNN comparison](results/kitti/pv_rcnn/validation/FINDINGS.md) |
| CLOCs on PointPillars, three seeds | Same KITTI validation; 96 optimization / 32 development frames | Moderate AP40: 79.86 / 76.27 / 78.76; baseline 79.13. Only one seed improves Moderate; all reduce Hard | [Seed diagnostics](results/kitti/clocs/seed_diagnostics/FINDINGS.md) |
| BEVFusion LiDAR vs camera–LiDAR | Availability-selected nuScenes validation subset: 1,231 samples, 31 scenes | mAP × 100: 64.21 vs 67.81; NDS × 100: 69.21 vs 71.15 | [Subset findings](results/nuscenes/bevfusion/validation_subset/FINDINGS.md) |
| BEVFusion mini pilot | 81 mini_val samples | mAP × 100: 54.50 vs 57.04; NDS × 100: 56.62 vs 57.62 | [Mini findings](results/nuscenes/bevfusion/mini/FINDINGS.md) |

## Evidence status — 8 October 2026

- PointPillars baseline parity checks reproduction against frozen results with the same evaluator. PV-RCNN native-versus-common checks evaluator agreement on the same predictions. These are different verification scopes.
- Original nuScenes subset evidence has been hash-verified and compact metrics, scripts, audits and summary curves imported. A CPU ground-truth replay reproduced both subset headline metrics exactly on 8 October; [replay evidence](results/nuscenes/bevfusion/validation_subset/replay_20261008/README.md) records the same-evaluator verification.
- The nuScenes subset is selected by sensor availability, not randomly, and is not full validation. BEVFusion compares two pretrained pipelines, not a controlled camera ablation. Mini/subset overlap is unchecked.
- CLOCs seed-29's Easy decline remains unexplained; full-AP40 score-blend testing, robustness, semantic fusion and comparable runtime across models remain pending. Validation was inspected iteratively.
- Full nuScenes validation and the dissertation are still in progress. Existing detector architectures are reused; the contribution is integration, evaluation and analysis.

## Repository layout

| Folder | Contents |
|---|---|
| `adapters/` | Detector interfaces |
| `fusion/` | CLOCs network, training, sampling and geometry |
| `configs/` | Split and configuration records |
| `scripts/` | Preparation, evaluation, auditing and plotting commands |
| `tests/` | Implementation checks |
| `results/kitti/` | Camera, depth, PointPillars, PV-RCNN and CLOCs studies |
| `results/nuscenes/` | BEVFusion mini and validation-subset studies |
| `results/audits/` | Shared split and environment checks |
| `docs/` | Technical guides, figure gallery and migration map |
| `requirements/` | Environment and source provenance |

Datasets, model checkpoints, virtual environments and bulk predictions are stored outside Git. See the [artifact policy](results/README.md#artifact-policy).
