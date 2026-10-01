# nuScenes mini pretrained BEVFusion pilot

Both official OpenMMLab checkpoints passed strict loading and were evaluated
on the same ordered 81 mini_val samples from two scenes using the pinned
MMDetection3D NuScenesMetric and nuScenes evaluator.

Camera–LiDAR minus LiDAR-only: mAP +2.5364 points and
NDS +0.9919 points (metrics multiplied by 100).

This compares two pretrained checkpoint pipelines. It is not a controlled
camera ablation, training experiment, or full nuScenes validation reproduction.
Trailer, construction vehicle and barrier have no annotated instances in this
mini_val subset. Preserve official absent-class and undefined-error outputs;
do not interpret them as observed class failures. Class counts printed by the
dataset are not necessarily the evaluator's post-filter ground-truth counts.
The dataset version was explicitly configured as v1.0-mini after the default
v1.0-trainval caused a pre-inference assertion failure. Prediction settings and
official evaluation class definitions were retained.

Figure: comparison.svg shows official mini_val mAP and NDS multiplied by 100,
with an axis from 0 to 100. Only these two scenes support this pilot comparison.
Checkpoint identities, source files, per-class metrics and setup logs are saved
with the archive receipt. The full ZIP retains all 162 per-sample predictions
and both official results_nusc.json files outside Git.
