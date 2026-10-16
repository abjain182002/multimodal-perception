# Multimodal Object Detection and Robotic Perception

## Camera challenges — 16 October 2026

[Seven-condition study](results/nuscenes/bevfusion/camera_challenges/20261016/README.md): blur, additive Gaussian noise and common extrinsic yaw are now evaluated on the same 600 frames. Fresh clean mAP/NDS ×100: **68.10/71.28**; stronger tested blur, noise and yaw produce mAP losses of **2.22/4.97/3.47 points**. [One compact three-panel figure](results/nuscenes/bevfusion/camera_challenges/20261016/figures/comparison.png). All original evidence retained; synthetic subset results do not establish physical-system robustness.

## LiDAR point-reduction challenge — 15 October 2026

[Matched 600-frame study](results/nuscenes/bevfusion/point_reduction/20261015/README.md): at 100% / 50% / 25% retention, LiDAR-only mAP ×100 is **64.31 / 60.56 / 51.04**, and fusion is **68.13 / 65.79 / 60.23**. Both degrade; fusion loses fewer points under this synthetic intervention. All 600 paired input records and deterministic selection hashes checked. [Report evidence/page plan](docs/report/EVIDENCE_AND_PAGE_PLAN.md).

## Image-darkening challenge — 15 October 2026

[Controlled 600-frame study](results/nuscenes/bevfusion/image_darkening/20261015/README.md): clean / gain 0.50 / gain 0.25 mAP ×100 = **68.1300 / 67.2352 / 64.5150**; NDS ×100 = **71.2886 / 70.8179 / 69.1982**. Same fusion checkpoint and frozen inputs; synthetic image brightness perturbation, not a real-night benchmark.

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
| BEVFusion full validation | 6,019 samples / 150 official validation scenes | mAP × 100: 64.43 vs 68.31; NDS × 100: 69.24 vs 71.14 | [Full findings](results/nuscenes/bevfusion/full_validation/FINDINGS.md) |
| BEVFusion LiDAR vs camera–LiDAR | Availability-selected nuScenes validation subset: 1,231 samples, 31 scenes | mAP × 100: 64.21 vs 67.81; NDS × 100: 69.21 vs 71.15 | [Subset findings](results/nuscenes/bevfusion/validation_subset/FINDINGS.md) |
| BEVFusion mini pilot | 81 mini_val samples | mAP × 100: 54.50 vs 57.04; NDS × 100: 56.62 vs 57.62 | [Mini findings](results/nuscenes/bevfusion/mini/FINDINGS.md) |

## Evidence status — 10 October 2026

- PointPillars baseline parity checks reproduction against frozen results with the same evaluator. PV-RCNN native-versus-common checks evaluator agreement on the same predictions. These are different verification scopes.
- Original nuScenes subset evidence has been hash-verified and compact metrics, scripts, audits and summary curves imported. A CPU ground-truth replay reproduced both subset headline metrics exactly on 8 October; [replay evidence](results/nuscenes/bevfusion/validation_subset/replay_20261008/README.md) records the same-evaluator verification.
- The nuScenes subset is selected by sensor availability, not randomly, and is not full validation. BEVFusion compares two pretrained pipelines, not a controlled camera ablation. Mini/subset overlap is unchecked.
- CLOCs seed-29's Easy decline remains unexplained; full-AP40 score-blend testing, robustness, semantic fusion and comparable KITTI runtime across models remain pending. Validation was inspected iteratively.
- Full nuScenes validation is complete: 6,019 samples / 150 scenes, with identical paired prediction/GT tokens. Original compact evidence and figures are preserved in the [full-validation folder](results/nuscenes/bevfusion/full_validation/README.md). The ZIP import verifies all 111 member hashes and stored metric arithmetic; a separate full-split replay remains pending. Existing architectures are reused; the contribution is integration, evaluation and analysis. The dissertation remains in progress.

## Repository layout

| Folder | Contents |
|---|---|
| `adapters/` | Detector interfaces |
| `fusion/` | CLOCs network, training, sampling and geometry |
| `configs/` | Split and configuration records |
| `scripts/` | Preparation, evaluation, auditing and plotting commands |
| `tests/` | Implementation checks |
| `results/kitti/` | Camera, depth, PointPillars, PV-RCNN and CLOCs studies |
| `results/nuscenes/` | BEVFusion full validation, subset and mini studies |
| `results/audits/` | Shared split and environment checks |
| `docs/` | Technical guides, figure gallery and migration map |
| `requirements/` | Environment and source provenance |

Datasets, model checkpoints, virtual environments and bulk predictions are stored outside Git. See the [artifact policy](results/README.md#artifact-policy).

## Annotated nuScenes cases

[Six case studies with camera crops and paired bird's-eye views](results/nuscenes/bevfusion/case_studies/20261013/README.md) illustrate benefits, regressions and shared failures. Captions separate observations from untested causal explanations.

## Camera availability experiment

[600-frame paired camera-masking study](results/nuscenes/bevfusion/camera_availability/20261014/README.md): intact six-view fusion scores 68.13 mAP / 71.29 NDS; front/back retained with four views masked scores 63.63 / 68.76; front masked scores 64.84 / 69.44. Same fusion weights and tokens; missing-view robustness, with six-view computation retained.


## Runtime, memory and architecture

[Three-session T4 benchmark](results/nuscenes/bevfusion/runtime/20261015/README.md): median-of-session-medians latency is 287.81 ms (LiDAR) versus 544.98 ms (fusion); maximum allocated GPU memory is 609.82 versus 2,231.80 MiB. Same 150-frame runtime subset; file loading excluded. [Architecture diagrams](docs/architecture/README.md) identify reused model components and project orchestration. [Hardware cost framework](results/cost_performance/README.md) records initial sources and unresolved price assumptions.




