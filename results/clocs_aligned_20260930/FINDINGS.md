# Aligned CLOCs findings — 30 September 2026

FOV-reduced training candidates were regenerated for the same 128 KITTI training frames. The original 96 optimization / 32 internal development split was fixed across three training seeds. Each run used all valid candidates and 12 epochs. Mean best development candidate AP was 0.870549, sample standard deviation 0.006397; PointPillars candidate AP was 0.777660. Every run selected epoch 12, so convergence remains unverified.

Seed 20260928 was selected by internal development candidate AP before this validation run. Full validation used this selected seed only; the three-seed statistics are not full-validation uncertainty estimates.

On 3,769 validation frames, the frozen PointPillars baseline reproduced all nine reference AP40 values exactly (maximum error 0.0). Aligned CLOCs 3D AP40 was 90.7684 Easy, 79.8616 Moderate and 74.8719 Hard. Relative to PointPillars: +0.4235, +0.7317 and -1.2846 AP points respectively. The mixed outcome must be reported, including the Hard decline.

Official PV-RCNN Moderate 3D AP40 was 84.3604 under the common evaluator, 4.4989 points above this aligned CLOCs run. This compares complete pipelines with different architectures and training budgets; it does not establish a general limitation of fusion. PV-RCNN is a two-stage detector, not PV-RCNN++ or Azitt.

Only 96 frames optimized the fusion head. Seed repeats do not answer the training-size question. Validation had already been inspected in earlier experiments, so these results are an iterative validation comparison rather than an untouched final test. The observed improvement over the earlier checkpoint does not isolate preprocessing as a causal effect because training environment and checkpoint selection also differ.

Sources: ap40.json, seed_comparison.json, study_protocol.json, validation_selection.json and each seed's history.json; PV-RCNN: ../pv_rcnn_kitti/common_ap40.json. SVG figures preserve editable vector output; CSV tables retain unrounded values.
