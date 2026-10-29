# Challenges, limitations and candidate mitigations — report writing guide

Updated: 29 October 2026. This guide links completed project evidence to discussion arguments. Candidate mitigations are proposed engineering/research responses, not demonstrated improvements unless an experiment explicitly establishes them. Snow and excessive-brightness extensions were discussed but have not been completed.

## How to use this material

- **Methods:** define the evaluated datasets, sensors, checkpoints, subsets and corruption protocols.
- **Results:** report only measured outcomes and their support.
- **Discussion:** explain implications, competing explanations, failures and trade-offs.
- **Limitations:** state conditions and deployment claims that the evidence does not establish.
- **Future work:** select testable mitigation proposals and specify how they would be evaluated.

Do not treat a plausible mechanism as an established cause, a proposed mitigation as implemented, or synthetic input degradation as validation of a whole weather condition.

## Completed environmental evidence

| Topic | Evidence available | Discussion argument and boundary | Candidate mitigation / next validation |
|---|---|---|---|
| Reduced brightness | [600-frame darkening experiment](../../results/nuscenes/bevfusion/image_darkening/20261015/README.md), image gains 1.00/0.50/0.25 | Quantifies sensitivity to encoded image-value reduction. It is not physical illumination modelling or direct-glare testing. | Investigate exposure-aware processing or illumination augmentation; evaluate against an unchanged baseline and held-out real illumination data. |
| Natural rain, night and their combination | [Completed scene-description group evaluation](NATURAL_WEATHER_GROUPS_2026-10-28.md), supported by selected cases | Groups are observational. Missing words do not establish dry/daylight conditions. Different class support, scenes and recording logs confound comparisons. | Collect/verify frame-level condition labels and rainfall intensity; use adequate independent recordings and matched or stratified analyses. |
| Synthetic fog | [Completed eight-run paired study](../../results/nuscenes/bevfusion/fog/20261029/completed_cpu/README.md) | Both pipelines degrade substantially. Fusion's higher aggregate score does not prevent class failures or imply smaller losses from baseline at every severity. Camera fog is heuristic; modality severities are not a common physical visibility distance. | Test real fog and additional corruption realisations; investigate weather-aware training or sensor-quality weighting and evaluate them rather than assuming a benefit. |
| Image blur/noise | [Camera challenges](../../results/nuscenes/bevfusion/camera_challenges/20261016/README.md) | Measured generic image degradations; not validated wind, snow, rainfall or vehicle-vibration models. | Investigate image-quality monitoring and augmentation; separately evaluate any restoration algorithm because it can also alter useful evidence. |
| Missing views and calibration errors | [Camera availability](../../results/nuscenes/bevfusion/camera_availability/20261014/README.md) and [yaw challenges](../../results/nuscenes/bevfusion/camera_challenges/20261016/README.md) | Show sensitivity to specific masking and misalignment interventions. Camera masking retains the six-view architecture; it does not establish a cheaper purpose-trained two-camera system. | Test calibration monitoring, sensor fault detection and missing-view training. A LiDAR-only fallback is a candidate system design requiring its own switching and failure evaluation. |
| Reduced point information | [100/50/25% point-retention study](../../results/nuscenes/bevfusion/point_reduction/20261015/README.md) | Quantifies matched point-subset sensitivity. It does not simulate all physical precipitation effects or replacement LiDAR hardware. Temporal sweeps are not separate physical LiDAR units. | Test quality monitoring and point-sparsity training; validate alternative hardware using its actual acquisition characteristics. |

## Geographic, terrain and deployment challenges

These entries describe **untested transfer risks**, not failures measured by this project. See [geographic/terrain applicability and primary references](GEOGRAPHIC_TERRAIN_APPLICABILITY_2026-10-29.md).

| Challenge | Limitation / defensible argument | Candidate mitigation and evidence required |
|---|---|---|
| Hills, slopes, banked roads and elevation changes | No terrain-gradient-stratified evaluation was performed. Road geometry and vehicle orientation may change the input distribution. The yaw calibration study is not a slope experiment. | Collect annotated routes with gradient/banking strata; verify pose and coordinate handling; evaluate unchanged models before terrain-aware adaptation. |
| Hill crests, bends and vegetation occlusion | Hidden objects may provide insufficient current sensor evidence. More modalities do not guarantee visibility around terrain. | Investigate temporal tracking and complementary sensor placement; evaluate occlusion and time-to-detection explicitly. Any speed or route policy belongs to separately validated downstream planning. |
| Delhi, Glasgow or another untested region | The nuScenes experiments concern specified Boston/Singapore data. Neither city-specific accuracy nor city-wide reliability is established. KITTI and nuScenes results do not by themselves form a controlled geographic-transfer study. | Define a target operating scope; obtain local annotations, object support and independent recording coverage; test the unchanged pipeline and preserve held-out data for any adaptation. |
| Different sensor rig or mobile robot | Input compatibility is not preserved merely by providing camera images and LiDAR points. Sensor specifications, intrinsics/extrinsics, timing and preprocessing may differ. | Audit input compatibility, calibration, synchronization and coordinate transforms; validate accuracy and latency on the target platform. |
| Terrain and unfamiliar hazards | The detector's taxonomy is not a complete obstacle or terrain-hazard specification. Object boxes do not establish cliff-edge, pothole, flood-depth or traversability estimation. | Add appropriately labelled hazard/terrain tasks and evaluate their outputs separately. |
| Physical deployment and real-time response | Benchmark detection and model-only timings do not establish an end-to-end safe robot or vehicle. | Measure acquisition, preprocessing, inference, tracking and actuation delays on the intended system; separately validate downstream behaviour. |

## Weather conditions remaining untested

| Condition | Appropriate report treatment | Required future validation |
|---|---|---|
| Real fog/haze | Synthetic fog motivates a limitation; it does not establish real-weather performance. | Annotated real-condition data, measured or well-defined visibility categories and independent recordings. |
| Snow, sleet and hail | Literature-supported gap. Do not describe an unexecuted snow proposal as a result. | Documented modality-specific simulation and/or real data, matched baselines, support and validation of assumptions. Snow does not automatically cover sleet/hail. |
| Direct sun glare and reflections | Darkening is not glare; excessive brightness was only proposed. | Real glare cases and/or a justified optical corruption protocol, with matched evaluation. |
| Wind | Blur/noise are generic effects, not a validated wind experiment. | Defined effects such as measured rig vibration or airborne material, with appropriate data and tests. |
| Dust, sand, smoke and pollution | Fog-model results cannot simply be relabelled as these conditions. | Relevant sensor/particle modelling or annotated real-condition data. |
| Storms, flooding and other disasters | Outside the current evaluated operating scope. | Task-appropriate datasets, geometry/hazard coverage and separate system validation; cosmetic overlays are insufficient. |

## Cross-cutting interpretation limits

- The LiDAR and fusion checkpoints are separately pretrained pipelines, not a controlled camera-component ablation.
- Selected visual examples illustrate mechanisms and outcomes; they do not estimate population-level performance.
- Repeated-frame annotations are not unique objects, and frames from a scene/log are correlated.
- The fog extension provides descriptive point estimates from one seeded realisation; no statistical significance, universal severity response or deployment-safety claim follows.
- Weather interactions and target-region generalisation remain distinct from individual corruption tests.
- Proposed mitigation effectiveness must be measured. Do not write that adaptation, exposure correction, tracking or fallback resolves a failure unless tested.

## Writing and assessment use

For each substantive discussion point, connect **question → evidence → alternative explanations → limitation → practical implication → testable next step**. Prioritise the strongest measured results over a long list of speculative hazards.

Use a concise environmental/terrain scope table in the main report, a few representative failure examples, and detailed audits in the repository or appendix. The same distinctions should be maintained in the poster and oral defence. The aim is a clear account of what the project establishes and what additional evidence deployment would require; experiment count does not determine the grade.

## Evidence index

- [Natural weather groups](NATURAL_WEATHER_GROUPS_2026-10-28.md)
- [Synthetic fog results and report placement](SYNTHETIC_FOG_RESULTS_2026-10-29.md)
- [Geographic and terrain applicability, with primary sources](GEOGRAPHIC_TERRAIN_APPLICABILITY_2026-10-29.md)
- [Illumination/weather case discussion](ILLUMINATION_WEATHER_DISCUSSION_2026-10-28.md)
- [Original visibility and point-support diagnostics](../../results/nuscenes/bevfusion/case_analysis/20261028/metadata_completion/README.md)
