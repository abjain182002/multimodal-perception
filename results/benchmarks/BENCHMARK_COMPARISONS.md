# Supervisor benchmark comparisons — 10 October 2026

## Purpose and evidence scope

This table separates measured project results from developer-reported benchmarks. It establishes reference context and documents reproduction differences; it does not claim state-of-the-art performance or an isolated camera/architecture effect. No inference was rerun for this table. All score differences below are absolute score points, not relative percentages.

## 1. nuScenes: checkpoint-linked reference comparison

Measured evaluation covers all 6,019 official validation samples across 150 scenes, with the official `detection_cvpr_2019` evaluator configuration. Both pipelines use the MMDetection3D BEVFusion implementation at commit `fe25f7a51d36e3702f961e198894580d83c4387b`, voxel size 0.075 m, and checkpoint-linked configurations. The upstream configuration specifies nine preceding LiDAR sweeps in addition to the current scan.

The reference values are the rounded MMDetection3D model-table results, not scores copied from another BEVFusion implementation or the hidden test leaderboard. Its configurations use validation annotations. Full training/evaluation-log equivalence has not been audited.

| Pipeline | Reference mAP ×100 | Measured mAP ×100 | Difference | Reference NDS ×100 | Measured NDS ×100 | Difference |
|---|---:|---:|---:|---:|---:|---:|
| LiDAR-only | 64.9 | 64.4301 | -0.4699 | 69.6 | 69.2353 | -0.3647 |
| Camera–LiDAR | 68.6 | 68.3099 | -0.2901 | 71.4 | 71.1352 | -0.2648 |

**Reference:** [MMDetection3D BEVFusion model table](https://github.com/open-mmlab/mmdetection3d/blob/fe25f7a51d36e3702f961e198894580d83c4387b/projects/BEVFusion/README.md). Checkpoint filenames end in `2628f933.pth` (LiDAR) and `5239b1af.pth` (camera–LiDAR); complete measured hashes are recorded in the [original merge audit](../nuscenes/bevfusion/full_validation/original_evidence/merge/merge_audit.json).

**Measured evidence:** [Full-validation findings](../nuscenes/bevfusion/full_validation/FINDINGS.md) and original metrics linked there. The reference has one-decimal precision; four-decimal differences are arithmetic against rounded reference values, not four-decimal estimates of upstream reproduction error.

**Interpretation:** Measured scores are lower than the checkpoint-linked reference by approximately 0.47/0.36 points (LiDAR mAP/NDS) and 0.29/0.26 points (camera–LiDAR). The reason is unresolved; do not attribute it to hardware, devkit versions, preprocessing or nondeterminism without investigation. Archive hashes and stored arithmetic were checked, but a fresh full-split ground-truth replay remains pending.

Within this project, camera–LiDAR exceeds LiDAR-only by **3.8798 mAP points** and **1.8998 NDS points**. This compares two pretrained pipelines, not a controlled camera ablation or equal-training-budget experiment.

## 2. KITTI: measured, same-split AP40 comparison

All rows below use the frozen 3,769-frame SECOND validation split, Car strict 3D IoU 0.70 and AP40. PointPillars/PV-RCNN are pretrained pipelines; the CLOCs head was trained with 96 optimization and 32 development frames. Preprocessing, backbone and upstream training recipes differ. Validation was inspected iteratively, so it is not an untouched final test set.

| Project pipeline | Easy AP40 | Moderate AP40 | Hard AP40 |
|---|---:|---:|---:|
| PointPillars | 90.3449 | 79.1299 | 76.1566 |
| PV-RCNN | 92.1047 | 84.3604 | 82.4830 |
| CLOCs + YOLO26x; Seed 20260928, epoch 12 | 90.7684 | 79.8616 | 74.8719 |
| CLOCs + YOLO26x; Seed 20260929, epoch 12 | 82.5581 | 76.2666 | 73.1530 |
| CLOCs + YOLO26x; Seed 20260930, epoch 12 | 89.8051 | 78.7616 | 73.4648 |
| CLOCs + YOLO26x; Seed 20260928, epoch 24 | 90.2610 | 79.4874 | 74.7797 |

**Measured sources:** [PointPillars/PV-RCNN comparison CSV](../kitti/pv_rcnn/validation/comparison.csv), [PV-RCNN findings](../kitti/pv_rcnn/validation/FINDINGS.md), and [CLOCs seed/epoch CSV](../kitti/clocs/seed_diagnostics/ap40_comparison.csv). The older 79.2542-Moderate CLOCs checkpoint remains historical evidence; it is not substituted for the later three-seed diagnostic.

PV-RCNN improves Moderate/Hard AP40 over PointPillars by 5.2306/6.3265 points. Only one epoch-12 CLOCs seed improves Moderate AP40, and all three reduce Hard AP40. Do not show only the favourable seed or claim general fusion failure.

Verification differs: PointPillars parity checks reproduce frozen baseline scores using the same evaluator. PV-RCNN native/common evaluator agreement checks agreement on the same predictions. Neither means every pipeline was independently cross-verified.

## 3. KITTI developer-reported context — not AP40 deltas

| Developer reference | Reported metric | Published score | Relationship to this project's AP40 |
|---|---|---:|---|
| OpenPCDet PointPillar | Car Moderate 3D AP11 / R11, validation | 77.28 | Different checkpoint/framework and recall sampling; no direct delta |
| OpenPCDet PV-RCNN | Car Moderate 3D AP11 / R11, validation | 83.61 | Recall sampling differs; no direct delta |
| MMDetection3D PointPillars, `d42d15ed` checkpoint | KITTI AP; detailed recall convention not specified in the README table | 77.6 | Same checkpoint link, but AP40 comparability unestablished; no direct delta |

**Sources:** [OpenPCDet pinned KITTI model table](https://github.com/open-mmlab/OpenPCDet/blob/233f849829b6ac19afb8af8837a0246890908755/README.md) explicitly labels its scores `Car@R11`; [MMDetection3D PointPillars table](https://github.com/open-mmlab/mmdetection3d/blob/fe25f7a51d36e3702f961e198894580d83c4387b/configs/pointpillars/README.md) labels its score only `AP`. We do not relabel either as AP40. Resolving the latter requires inspecting its linked evaluation log and evaluator convention.

KITTI's official benchmark switched from 11 to 40 recall positions in October 2019. Its public leaderboard evaluates the separate hidden test set, whereas this project's measurements use a split of the annotated training set. Neither AP11 values nor hidden-test scores are directly interchangeable with the local validation AP40 table. Source: [KITTI benchmark documentation](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).

## 4. Measurement definitions and presentation rules

- **KITTI AP40:** class-specific precision–recall summary at 40 recall positions; this table is Car 3D IoU 0.70 with Easy/Moderate/Hard difficulty definitions.
- **nuScenes mAP:** average AP across ten classes and 0.5/1/2/4 m ground-plane centre-distance thresholds. It is not KITTI 3D-IoU AP40.
- **NDS:** combines mAP with translation, scale, orientation, velocity and attribute errors; not simply a percentage of correctly detected objects.
- Always label validation versus test, model/checkpoint, modality, class coverage and score convention. Do not merge KITTI and nuScenes scores into a single ranking.
- Runtime: the existing PV-RCNN T4 benchmark is measured, but a common cross-model runtime protocol is pending. Do not compare paper FPS with project Mac/T4 timings as if hardware and timing boundaries matched.

Metric definitions: [nuScenes developer documentation](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/README.md) and KITTI documentation above. Hashes of the inspected primary-source text are in [sources.json](sources.json); exact measured values are in the accompanying CSVs.

## 5. Case-study work remaining

Select four to six paired nuScenes examples from saved predictions: fusion helps; both models succeed; fusion underperforms; both models fail. Record sample tokens, selection criteria, thresholds, matching rules and limitations. Render ground truth and both predictions on identical camera/BEV views, restoring only selected sensor files. This document does not claim case-study visuals are complete, and selected examples cannot establish aggregate performance.

Existing PointPillars diagnostic frames 002744 (distant success), 004096 (localisation near miss), and 001199 (high-confidence unmatched diagnostic) are recorded in its [evidence manifest](../kitti/pointpillars/pointpillars_evidence_manifest.json). They can support a separate KITTI demonstration; the unmatched example is not automatically an official KITTI false positive.

## Supervisor takeaway

The completed full-validation BEVFusion comparison shows a positive aggregate camera–LiDAR advantage while remaining slightly below its checkpoint-linked reference scores. KITTI shows a stronger LiDAR backbone outperforming this small-data fusion implementation. These observations motivate case studies and diagnostic analysis; they do not isolate a dataset effect, establish statistical significance, or demonstrate robot deployment.
