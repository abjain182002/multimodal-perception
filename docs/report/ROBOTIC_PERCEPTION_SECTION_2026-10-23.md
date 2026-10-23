# From camera-LiDAR detection to mobile-robot obstacle alerts

## 1. Purpose and connection to the project

This software-only dissertation evaluates multimodal object detection and its relevance to robotic perception. A detector's benchmark accuracy does not directly establish whether its outputs improve a robot's decisions. To examine that connection, an offline demonstrator was implemented that transforms preserved three-dimensional detections into a robot-relative frame and tests whether their footprints intersect a predefined forward corridor. The resulting alert is a perception output that a future navigation system could consume; no motion controller or physical robot was evaluated.

The research question was whether the camera-LiDAR pipeline produced fewer missed corridor alerts than the LiDAR-only pipeline on identical recorded frames. This comparison reuses the project's pretrained models and saved predictions. It evaluates two complete pipelines, rather than isolating the causal effect of adding cameras through controlled retraining. Its contribution is an implemented perception-to-decision interface and an assessment of the evidence needed to support downstream claims.

A six-case feasibility pilot established the coordinate transformation and intersection procedure, but none of its primary corridors contained an annotated obstacle. The analysis was therefore extended to the existing frozen 600-frame subset, containing four frames from each of 150 nuScenes validation scenes. The original six illustrative cases were not pooled with the larger study. No new detector inference or outcome-based frame selection was performed for the 600-frame analysis [1].

## 2. Experimental design

A hypothetical robot measuring 1.0 m long by 0.8 m wide was centred at the recorded ego origin. Adding a 0.1 m lateral margin on each side gave a 1.0 m-wide alert corridor. The primary corridor extended 5 m from the assumed robot front: x from 0.5 to 5.5 m and y from -0.5 to 0.5 m. Forward lengths of 2 and 10 m were predeclared sensitivity settings. The primary confidence threshold was 0.30, with additional thresholds of 0.10 and 0.50. These dimensions are illustrative design assumptions, not measured braking or stopping distances.

Coordinates use x forward, y left and z upward, consistent with the ROS body-frame convention [2]. Global prediction corners were transformed using the inverse recorded ego-to-global matrix. Ground-truth corners were reconstructed from the stored LiDAR-frame centre, length, width, height and yaw, then transformed using the LiDAR-to-ego matrix. The box convention was checked against the pinned MMDetection3D conversion code [3]. The convex hull of the eight corners projected onto the horizontal plane formed each footprint. Boundary contact counted as corridor intersection; no vertical clearance or motion prediction was applied.

<!-- PAGEBREAK -->

## 3. Alert outcomes and measured results

A model raised an alert when at least one predicted footprint above the confidence threshold intersected the corridor. The annotation reference used the corresponding intersection rule. A missed alert meant no predicted alert in an annotation-positive frame. An unreferenced alert meant a predicted alert in an annotation-negative frame. The latter term avoids treating incomplete annotations as proof of physically clear space. These are binary frame-level outcomes: a correct alert does not necessarily demonstrate correct object association.

**Table 1.** Fixed-corridor results at confidence >=0.30 on all 600 frames. Positive and negative refer to the annotation-based corridor reference. Values are counts, except the percentages shown in parentheses. Existing predictions were reused; the analysis ran on CPU [1].

| Length | Positive / negative frames | LiDAR misses | Fusion misses | Unreferenced alerts: LiDAR / fusion |
|---|---|---|---|---|
| 2 m | 1 / 599 | 1 / 1 | 1 / 1 | 0 / 0 |
| 5 m (primary) | 2 / 598 | 1 / 2 | 1 / 2 | 0 / 1 |
| 10 m | 57 / 543 | 7 / 57 (12.28%) | 5 / 57 (8.77%) | 0 / 0 |

![Combined corridor results](../../results/robotics/mobile_robot_corridor600/20261023/results/comparison.png)

**Figure 1.** Annotation-positive frame counts, missed-alert rates and unreferenced-alert rates for the three tested lengths at confidence >=0.30. The corridor is 1.0 m wide. Lines connect discrete settings, without establishing intermediate behaviour. Point estimates are shown; scene-bootstrap uncertainty is discussed below. The primary 5 m corridor has only two positive frames, limiting comparative inference. This is offline road-scene analysis, not physical robot validation [1].

At 5 m, both pipelines detected one positive frame and missed the other; fusion additionally produced one unreferenced alert. At 10 m, fusion recovered three alerts that LiDAR missed but lost one supplied by LiDAR, giving a net reduction of two misses. Consequently, the longer-corridor result contains both improvements and regressions rather than a uniform fusion benefit.

<!-- PAGEBREAK -->

## 4. Uncertainty and critical interpretation

The 57 annotation-positive frames at 10 m came from 29 scenes. To account for dependence between frames from the same scene, uncertainty was estimated using 2,000 paired scene-bootstrap resamples with seed 20261023. Each resample selected 150 scenes with replacement, retaining all four selected frames and both models for every sampled scene. The percentile 95% interval for fusion-minus-LiDAR missed-alert rate at 10 m and threshold 0.30 was -11.11 to +3.57 percentage points, around an observed difference of -3.51 points. Because the interval includes zero, these data do not establish a reliable fusion advantage [1].

The primary 5 m result is even more constrained. Only 1,743 of the 2,000 resamples contained an annotation-positive frame; undefined missed-alert ratios were excluded and counted. Each model's missed-alert interval spanned 0-100%. Their paired difference was always zero among defined resamples because their observed primary misses were identical. That degenerate interval does not establish general equivalence. Likewise, a bootstrap interval of zero for an outcome never observed in this sample cannot establish a zero real-world failure rate.

The threshold checks further qualify the interpretation. At 10 m and threshold 0.10, both models missed three of the 57 positive frames. At threshold 0.50, LiDAR missed eight and fusion missed five. Thus the apparent advantage depends on the chosen operating threshold. These descriptive checks do not identify an optimal deployment threshold: optimisation would require separate development data and an explicit trade-off between missed and unreferenced alerts.

The experiment also shows why overall detection accuracy and task-specific alert performance must be discussed separately. Detections outside a narrow forward corridor do not change its binary alert. Conversely, a small footprint or score change near a corridor boundary can change an alert without materially changing aggregate benchmark performance. The additional fusion alert in scene-0563 was unreferenced at 5 m, while the same frame was annotation-positive at 10 m. This is an observed change in reference status; the responsible image features and physical cause have not been established.

## 5. Validity and reproducibility

All uploaded file hashes were verified. The two prediction token sets matched the 600 ordered annotation tokens, and the preserved selection confirmed 150 scenes with four frames each. Geometric checks covered known rotations, containment, separation, boundary contact and coordinate roundtrips. These checks support implementation consistency but do not replace validation against a physical robot or independently surveyed geometry.

The supplied annotations retain yaw-only ground-truth orientations and lack the metadata required for the SDK's bike-rack exclusion. The analysis retained known classes with positive combined LiDAR/radar point counts and applied the declared class-centre ranges. It is therefore an annotation-based diagnostic, not an exact official SDK replay or a new mAP/NDS evaluation. Unlabelled or excluded obstacles, geometry approximation and absence of temporal reasoning limit interpretation. Bootstrap intervals capture scene-sampling variation only, conditional on these frames, annotations and checkpoints.

<!-- PAGEBREAK -->

## 6. Proposed factory application and remaining requirements

A relevant future application is a mobile robot transporting components between factory workstations. In that proposed architecture, camera-LiDAR detections would provide semantic and spatial information to an obstacle-monitoring component. A navigation system would then combine that information with localisation, a map, obstacle observations and motion constraints. The present demonstrator implements only the static detection-to-corridor-alert connection; it does not implement that complete architecture.

Factory transfer would require evaluation with the intended robot's sensor mounting, synchronisation and calibration. It would also require data covering the actual operational environment, including workers, pallets, low shelving, cables and machinery. These examples are proposed evaluation requirements, not classes shown to be recognised by the current road-scene checkpoint. An alert based only on supported semantic classes cannot establish general obstacle coverage. Temporal tracking, motion planning, stopping behaviour and the response to missing or delayed sensor information would require separate tests.

The project's preserved robustness experiments provide useful motivation for those tests: they examine camera absence, brightness reduction, blur, noise, calibration perturbation and point sparsity. They do not establish factory performance, and the corridor analysis reported here uses nominal predictions rather than replaying every corruption condition. This distinction prevents synthetic detector robustness from being presented as demonstrated closed-loop robot robustness.

Compute measurements also constrain the proposed application. The separate single-T4 benchmark recorded median-of-session-medians durations of 287.81 ms for LiDAR-only and 544.98 ms for fusion, at batch size one and FP32, with file loading excluded [4]. At an assumed speed of 0.5 m/s, those durations correspond arithmetically to approximately 0.144 m and 0.272 m of travel. These illustrative distances omit acquisition, loading, downstream processing and braking; they are neither stopping-distance estimates nor measurements of robot motion. Deployment suitability would depend on the complete sensing-to-actuation delay and task requirements.

## 7. Contribution and scope

The robotics contribution is a reproducible software interface connecting preserved 3D detections to a defined spatial alert, together with a critical assessment of its outcomes. The primary comparison is limited by obstacle scarcity. The longer predeclared corridor shows a small, uncertain difference and identifiable gains and regressions. These results support a bounded claim of offline robotic-perception analysis, not guaranteed collision avoidance, factory readiness or a novel detection architecture. Further confirmatory work should use independently specified obstacle-rich robot trajectories and suitable annotations rather than tuning corridors to these outcomes.

## References and evidence

[1] Project evidence, *600-frame mobile-robot corridor study*, 23 October 2026. [Methods, protocol, code, frame results and uncertainty](../../results/robotics/mobile_robot_corridor600/20261023/README.md). Analysis preserved at commit `a9cdd68490dd22b70290d081113c823b8f529dfe`.

[2] Foote, T. and Purvis, M., *REP-103: Standard Units of Measure and Coordinate Conventions*. [Open Robotics](https://reps.openrobotics.org/rep-0103/). Accessed 23 October 2026.

[3] OpenMMLab, *nuScenes dataset conversion*, source commit `fe25f7a51d36e3702f961e198894580d83c4387b`. [Converter](https://github.com/open-mmlab/mmdetection3d/blob/fe25f7a51d36e3702f961e198894580d83c4387b/tools/dataset_converters/nuscenes_converter.py). Accessed 23 October 2026.

[4] Project evidence, *BEVFusion runtime and GPU memory*, 15 October 2026. [Timing boundary and measured results](../../results/nuscenes/bevfusion/runtime/20261015/README.md).
