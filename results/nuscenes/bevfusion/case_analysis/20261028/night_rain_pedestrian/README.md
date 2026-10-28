# Additional positive night/rain pedestrian case: frozen preparation plan

**Status: protocol and CPU preparation utility ready; actual selection, input recovery and visual/diagnostic analysis remain pending. No new result is claimed.**

The original night study contains one positive car frame and one annotation-negative night/rain frame. This bounded amendment adds one annotation-positive pedestrian frame while retaining both original examples. It addresses that specific evidence gap before dissertation writing resumes; it is not a new broad benchmark or training campaign.

## Selection fixed before predictions

The [protocol](selection_protocol.json) defines a deterministic rule:

1. Restrict to official validation scenes explicitly labelled night/nighttime and rain/rainy/raining in scene descriptions. These remain scene-level context, not measured frame-level weather intensity.
2. Require six camera keyframes and the current LiDAR keyframe in the preserved, hash-verified part-10 backup. This is an availability restriction, not random sampling.
3. Require an annotated pedestrian in one of the four official detection categories, with combined annotated LiDAR/radar support greater than zero and ego XY distance strictly below 40 m at the LiDAR keyframe pose.
4. Use the first scene alphabetically containing an eligible frame. Within it, choose the eligible frame nearest the middle timestamped validation frame; break ties by sample token.
5. Preserve the protocol, candidate audit and frozen frame selection on Drive before opening either model's predictions. If there is no eligible frame, stop without silently relaxing the rule.

Later visual selection uses the nearest eligible pedestrian with a valid camera projection, and selects the camera with the largest clipped GT projection. This also avoids model-outcome selection. Both models share the image/crop. Retain the existing local filters, one-to-one XY <2 m matching and thresholds 0.10/0.30/0.50; 0.30 remains primary. No threshold optimisation is planned.

## Preparation and required handoff

Run [prepare_inputs_colab.py](prepare_inputs_colab.py) in CPU Colab with the existing Drive mounted. It needs the original metadata backup, preserved part-10 keyframes and both full-validation prediction files at the already recorded paths. All source hashes are checked. It streams original metadata, applies the frozen selection, extracts only seven sensor keyframes and the selected saved predictions, then preserves and verifies a compact ZIP and receipt on Drive. The JSON-array reader uses a tested UTF-8 streaming reader compatible with streaming TAR members.

The ZIP must then be supplied for image review and diagnostic analysis. Preparation success does not establish a model match, pedestrian visibility in a chosen crop, or positive weather robustness. No additional sweeps, package installations, model loading or GPU inference are required for this saved-prediction analysis.

Expected preparation marker: `POSITIVE_NIGHT_RAIN_INPUTS_AND_DRIVE_COPY_PASS`. [Local synthetic preparation checks](preparation_checks.json) cover selection boundaries, tie-breaking, missing-file exclusion and a multi-chunk Unicode streaming-TAR fixture; actual dataset execution remains pending.

## Completion boundary

The addition is complete only after source/input checks, original-image review, confirmation that a pedestrian is eligible/projectable, paired diagnostic matching, a compact figure/table, bounded discussion and preservation. Keep any failure or unexpected outcome; do not search for a more favourable prediction. If the selected inputs cannot support the planned pedestrian view, record that limitation before agreeing any amendment.

One selected positive example will complement the existing cases. It cannot establish representative night/rain accuracy, independent recording conditions, weather causality or physical robot safety. After this bounded addition, the intended next work is dissertation integration rather than an expanded experiment list.
