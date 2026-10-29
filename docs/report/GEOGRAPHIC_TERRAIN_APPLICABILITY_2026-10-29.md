# Geographic transfer, terrain and operating limits — 29 October 2026

## Central argument

The project evaluates object detection on specified benchmark data and controlled corruptions; it does not validate operation in every city, terrain or weather condition. A pipeline may accept correctly formatted inputs from a new region and produce boxes without preserving detection accuracy. Detection results also do not establish safe navigation, braking or manipulation.

The official nuScenes documentation identifies Boston Seaport and Singapore's One North, Queenstown and Holland Village as collection locations [1]. Neither Delhi nor Glasgow nor a dedicated mountainous deployment was evaluated here. This does not imply that every benchmark road is flat. No terrain-gradient strata or mountain-route performance measurements were constructed.

## Region and terrain discussion matrix

The following are prospective failure mechanisms and validation needs, not measured failures of the implemented models.

| Setting or change | Why results might change | Existing evidence and boundary |
|---|---|---|
| Hills, steep gradients and banked roads | Changed road/vehicle orientation, visibility at crests and bends, and objects at different elevations may change the inputs and detection geometry. | No controlled terrain study. Camera yaw perturbations test miscalibration; they do not simulate road gradients. |
| Rural or mountain routes | Different scenery, road layout, vegetation, object frequencies and potentially unfamiliar hazards may differ from the benchmark distribution. | Urban benchmark results do not establish rural or mountain-route accuracy. |
| New city or country | Traffic mix, object appearance, road markings, layouts and acquisition conditions can differ. | Natural rain/night groups describe conditions within the evaluated dataset; they are not Delhi/Glasgow transfer tests. |
| New weather or combinations | Visibility and measurement degradation may differ from the tested effects and severity scales. | Rain/night observations and synthetic fog support bounded claims. Real fog, glare, snow, wind, dust and disaster performance remain unvalidated. |
| Different vehicle, robot or sensor rig | Camera intrinsics, sensor placement, point characteristics, timing and platform movement may differ. | Exact sensor formatting, calibration, synchronization and coordinate transforms must be checked before model use. Existing corruption studies do not validate a replacement rig. |
| Unfamiliar objects and terrain hazards | Fixed detection taxonomies do not cover every obstacle or surface hazard. | Object detection alone does not establish pothole, cliff-edge, landslide, flood-depth or traversability estimation. |

A CVPR study identifies ego-car pose changes relative to the ground, including road slope/smoothness, as a difficulty in monocular 3D detection [2]. This motivates asking the terrain question; it does not quantify slope sensitivity of this project's LiDAR/fusion checkpoints.

## What the project can defend

- Specified pretrained pipelines were evaluated on stated benchmark splits and matched input challenges.
- The experiments identify improvements, regressions and severe degradation under some tested inputs.
- KITTI and nuScenes broaden the project evidence, but comparing different pretrained models on different datasets is not a controlled geographic-transfer experiment.
- Algorithmic portability is a design property; geographic accuracy and deployment safety need additional evidence.
- The completed scope is sufficient to investigate the stated bounded research questions. It is not evidence of all-weather or all-terrain operation.

## Validation required before a new deployment

1. Specify the intended region, route types, road-gradient range, operating conditions, sensors and required object/hazard coverage.
2. Verify intrinsics, extrinsics, coordinate conventions, time alignment, point features, preprocessing and model input limits. Preserve original model assumptions rather than assuming a new rig is compatible.
3. Collect independently annotated target-region data covering slopes, bends, occlusion, distances and relevant environmental combinations.
4. Evaluate the unchanged model first using a common metric/matching protocol; report support and uncertainty at scene or recording level as appropriate.
5. Keep any adaptation/training data separate from a held-out target evaluation. Re-evaluate after changes.
6. Validate the downstream robot/vehicle system separately. Object-detection accuracy alone is insufficient for safety claims.

A single rigid point-cloud rotation or image warp should not be presented as real mountainous-terrain validation: it does not recreate changed viewpoint, newly hidden/revealed surfaces, sensor acquisition or target-region distribution.

## Dissertation placement

Introduce the geographic and sensor scope in Methods. In Discussion, distinguish measured robustness from untested geographic/terrain transfer. In Limitations/Future Work, identify the target-region validation above. Reuse this boundary in the poster and oral defence.

The completed fog study and natural weather analysis remain unchanged. Synthetic snow and excessive-brightness extensions were discussed but have not been executed or accepted as completed results.

## Sources

[1] nuScenes, official dataset documentation, Data Collection / Scene planning: https://www.nuscenes.org/nuscenes . Retrieved 29 October 2026. Supports acquisition locations and benchmark scope.

[2] Zhou et al., *Monocular 3D Object Detection: An Extrinsic Parameter Free Approach*, CVPR 2021, pp. 7556–7566: https://openaccess.thecvf.com/content/CVPR2021/html/Zhou_Monocular_3D_Object_Detection_An_Extrinsic_Parameter_Free_Approach_CVPR_2021_paper.html . Terrain/pose motivation is specific to that paper's monocular setting.

Project evidence: [completed fog results](../../results/nuscenes/bevfusion/fog/20261029/completed_cpu/README.md), [natural weather groups](NATURAL_WEATHER_GROUPS_2026-10-28.md), and [camera challenges](../../results/nuscenes/bevfusion/camera_challenges/20261016/README.md).
