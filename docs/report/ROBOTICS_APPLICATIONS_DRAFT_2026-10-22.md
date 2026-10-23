> **Report integration update (23 October 2026):** Use the [four-page robotics section](ROBOTIC_PERCEPTION_SECTION_2026-10-23.md) and [PDF](ROBOTIC_PERCEPTION_SECTION_2026-10-23.pdf), which incorporate the completed 600-frame study. This earlier draft is retained as project history.

# Application to factory mobile robots and manipulation

Status: discussion draft, 22 October 2026. Intended allocation: approximately 3–4 dissertation pages after formatting, including one compact table and an optional combined architecture figure. Page count is not yet typeset or verified. This section interprets preserved perception experiments. A separate [simulated robotic-arm pilot](ROBOT_ARM_PILOT_2026-10-22.md) now adds handling-control evidence; it does not use the perception model. No physical robot deployment or new obstacle-zone evaluation has been completed.

## 1. What the project contributes to a robot

The project evaluates camera and LiDAR object-detection pipelines, their sensitivity to controlled input changes, and their computational requirements. These are perception capabilities that could supply information to a robot's decision-making system. They do not constitute a complete autonomous robot. A deployed system would additionally require localisation, time synchronisation, coordinate transforms, planning, control and application-specific validation. Manipulation would introduce grasp estimation and contact feedback as further requirements.

The most direct industrial application to discuss is a mobile robot transporting material around a factory. Detecting and locating people or other relevant obstacles could inform a planner, while class labels could support different behaviour around people and equipment. However, the present KITTI and nuScenes experiments use road-scene data. Their measured accuracy cannot be transferred numerically to factory environments. Pallets, shelving, workpieces and other factory-specific obstacles require suitable data, labels and evaluation. An object detector also cannot establish that space is free merely because it returns no recognised object.

Factory applicability is therefore a transfer hypothesis supported by engineering analysis, rather than an experimentally demonstrated result. The dissertation's defensible claim is that the work identifies perception tradeoffs and failure mechanisms relevant to a future robotic integration. Claims of successful industrial navigation, safe operation or dexterous manipulation would require additional evidence.

## 2. Connecting preserved results to industrial questions

The existing case analysis provides a concrete reason to consider downstream decisions. At the diagnostic confidence threshold of 0.30, fusion matched 33 of 46 filtered objects in scene-0332, compared with 22 for LiDAR-only. In scene-0562 the ordering reversed: fusion matched 14 of 38, compared with 21 for LiDAR-only. These are outcome-selected examples, not estimates of failure prevalence. Their value is showing that aggregate benefits coexist with individual regressions.

The saved analysis further distinguishes a rejected low-score candidate from the absence of a nearby candidate. A robot acting on thresholded detections could consequently receive different obstacle information even when both networks place a plausible box near an object. Lowering the threshold can recover matches, but also introduces false positives. For a mobile robot, the corresponding operational tradeoff could be missed obstacles versus unnecessary slowing or stopping. The project has not yet measured those operational outcomes, so this connection remains an interpretation.

The robustness experiments support a similarly bounded discussion. Camera masking tests availability of selected views within the existing model interface. Image darkening, blur and noise probe specific synthetic image changes; yaw perturbation tests the chosen extrinsic intervention; matched point reduction probes uniform density reduction. None establishes real factory performance under dust, vibration, glare or physical sensor failure. These studies identify useful validation questions for a future installation, rather than substituting for that validation.

| Factory scenario | Relevant project evidence | Additional evidence or capability required |
|---|---|---|
| Mobile material transport near people | Paired detections, individual misses and false positives | Factory data, tracking, localisation, planning and system-level validation |
| Camera degradation or obstructed views | Frozen camera-availability and image-corruption studies | Real illumination, obstruction and sensor-health tests |
| Changes in sensor mounting | Controlled yaw intervention | Installation-specific calibration and drift monitoring |
| Embedded perception on a moving platform | Measured T4 inference time and memory | Target-computer measurements and complete sensing-to-action latency |
| Picking fragile or narrow objects | General perception architecture only | Object-specific perception, grasp planning, gripper and contact control |

## 3. Timing and a proposed mobile-robot interface

The preserved runtime table reports median-of-session-median inference times of 287.81 ms for LiDAR-only and 544.98 ms for fusion on a single Tesla T4. These measurements used 150 samples and three sessions per model; the accompanying accuracy values came from a separate 600-frame evaluation. The inference boundary excludes loading and must not be described as full sensing-to-action latency. Peak allocated GPU memory was approximately 609.82 MiB and 2231.80 MiB respectively.

An illustrative calculation makes the timing relevant to motion. At an assumed constant speed of 0.5 m/s, travel during those inference intervals alone would be approximately 0.144 m and 0.272 m. These distances are calculated as speed multiplied by inference duration. They are neither stopping distances nor validated clearance requirements: sensor age, communication, planning, actuation and braking are excluded. Actual robot design must measure the complete timing chain on the selected hardware.

A proposed integration would transform detections into the robot body frame and pass time-stamped obstacle information to a planning interface. Navigation software such as Nav2 illustrates a distinct downstream function: its Collision Monitor can use sensor observations and configured regions to request slowing or stopping [1]. This project has not implemented that component or an adapter for its detections. A conceptual architecture must label perception as evaluated and navigation/control as proposed.

An optional CPU-only extension could compare alerts obtained from saved predicted boxes with alerts obtained from ground-truth boxes in a predefined hypothetical corridor. Before computing results, it should freeze the footprint, corridor geometry, coordinate transform, confidence rule and alert definition. It would require complete boxes and frame poses, not just rendered figures. Selected cases would support illustration only; annotated detection classes would not provide exhaustive obstacle ground truth. This extension remains pending and should not be cited as a completed experiment.

## 4. Why picking an egg differs from picking a pen

Object recognition and gripping answer different questions. Perception estimates what an object is and where it is; grasp planning selects contact locations and orientation; the controller executes the motion and responds to contact. An egg and a pen can require different contact geometry, finger spacing, approach direction and acceleration even when both are light. Object identity alone is insufficient to determine a safe grip force.

For an egg, a plausible design uses compliant contact surfaces, a gentle approach and distributed contact to avoid concentrated loading. For a pen, accurate alignment with the narrow body and a stable pinch may be more important. The pen does not automatically require a larger force. Friction, material, geometry and the intended motion affect both cases, and no universal force setting is established here. These examples illustrate requirements rather than tested grasp recipes.

Real gripper interfaces expose several independent controls. Franka's documented grasp command accepts width, speed, force and width tolerances [2]. A configured force is not evidence that an unfamiliar object can tolerate it. Contact feedback can come from suitable force/torque sensors, joint sensing or tactile fingertips; their capabilities differ. GelSight research demonstrates sensing of shear and slip, which can help identify when a held object starts moving relative to the fingers [3].

Force control aims to regulate contact force. Impedance control instead specifies how compliant or resistant the robot should behave during displacement and interaction, commonly through stiffness and damping. Motion and force objectives can also be combined for different task directions [4]. These concepts help explain how a robot can approach an object, establish contact and adjust its behaviour instead of blindly continuing a position command. They are separate from the choice of an object-detection network.

The current road-scene detector has not been trained or evaluated for eggs, pens, grasp poses, contact forces or object fragility. Extending it to picking would require appropriate close-range sensing and object annotations, a suitable gripper, motion planning and feedback control, followed by physical evaluation. Grasp success, slip, damage, placement accuracy and cycle time would be relevant outcomes. Existing nuScenes mAP and NDS cannot measure those outcomes.

## 5. Scope and contribution

The factory-robot discussion connects measured perception behaviour to application requirements. The separate robotic-arm pilot adds a bounded simulation example of handling control, while mobile-robot integration and physical manipulation remain proposed extensions. The discussion and pilot should share the existing 3–4-page allocation, using a single composite figure and compact table. The simulation does not establish end-to-end performance of the project's perception pipeline.

## Evidence and references

- Project case evidence: [paired case-analysis draft](CASE_ANALYSIS_DRAFT_2026-10-20.md) and [saved analysis](../../results/nuscenes/bevfusion/case_analysis/20261020/README.md).
- Project timing evidence: [accuracy and compute matrix](../../results/nuscenes/bevfusion/runtime/20261015/tables/accuracy_compute_matrix.csv). Values checked against repository commit `bc04b40d1ad65b57d81f234e2d6a39a2334a2b1c` on 22 October 2026.
- [1] Nav2, [Using Collision Monitor](https://docs.nav2.org/rolling/tutorials/general_tutorials/using_collision_monitor/using_collision_monitor/). Software documentation; checked 22 October 2026.
- [2] Franka Robotics, [libfranka 0.20.2 Gripper API](https://frankarobotics.github.io/libfranka/0.20.2/classfranka_1_1Gripper.html). Command-interface evidence, not an egg-handling validation; checked 22 October 2026.
- [3] Yuan et al., [Measurement of Shear and Slip with a GelSight Tactile Sensor](https://people.csail.mit.edu/yuan_wz/GelSight1/ICRA15_2740_FI.pdf), ICRA 2015. Primary paper; retrieved 22 October 2026.
- [4] Lynch and Park, *Modern Robotics*, [Chapter 11: Robot Control](https://modernrobotics.northwestern.edu/chapters/chapter11/). Authors' textbook resources on motion, force and impedance control; checked 22 October 2026.


## Mobile-robot follow-up — 23 October 2026

The direct camera–LiDAR robotics connection has now been implemented as an [offline corridor-alert replay](../../results/robotics/mobile_robot_corridor/20261023/README.md) using the six preserved nuScenes cases and their recorded ego poses. The fixed 5 m corridor contains no filtered annotated obstacle in any of these frames; at the predeclared 10 m sensitivity setting, one frame contains an obstacle and both models alert. This establishes the coordinate/decision interface but does not demonstrate a fusion benefit, navigation performance or factory deployment. The six outcome-selected cases are inadequate for a reliability comparison. A broader saved-prediction evaluation remains pending. The separate arm pilot is supplementary and does not use this detector.

## Completed 600-frame follow-up — 23 October 2026

The broader saved-prediction analysis is now [complete and preserved](../../results/robotics/mobile_robot_corridor600/20261023/README.md), superseding the earlier pending status. At score 0.30, the primary 5 m corridor contains only two annotation-positive frames: both models alert on one and miss one; fusion has one additional unreferenced alert. The predeclared 10 m sensitivity condition has 57 positive frames, with seven LiDAR misses versus five fusion misses. Fusion gains three alerts and loses one. The fusion-minus-LiDAR missed-alert difference is −3.51 percentage points, with a paired 2,000-scene-resample percentile interval of −11.11 to +3.57 points. These results do not establish a reliable fusion advantage.

The supplied annotation file retains yaw-only GT orientation and lacks bike-rack metadata. This follow-up therefore uses an explicit annotation-based reference rather than exact official SDK filtering; it is not directly pooled with the six-case pilot. It demonstrates an offline perception-to-alert interface and the need to assess positive-case coverage before claiming downstream benefit. Physical robot, stopping and factory-domain validation remain unperformed. One composite and a compact table are sufficient for the report.
