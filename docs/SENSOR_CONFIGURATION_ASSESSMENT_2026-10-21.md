# Sensor configurations: evidence, feasibility and experiment decisions

Assessment date: 21 October 2026. Project evidence inspected at commit `5f5c3931afffda42abf381170be6592eec005d4e`.

## Research question and present decision

Which combination of cameras and physically distinct LiDAR sensors provides useful detection accuracy, coverage and resilience for its acquisition and computing cost?

This question requires more than naming a detector. The comparison must specify physical sensor count and placement, time history, trained model, evaluation region, classes, metric and computing protocol. A camera view masked at inference is an intervention on the existing model; a system built and trained for fewer cameras is a different experiment.

The assessment is complete at the documentation/source-inspection level. It establishes existing evidence, identifies concrete pretrained camera-only paths and records why a physical multi-LiDAR run is deferred. It does not report any new inference or certify that the candidate models run in the project's environment. Financial scenarios remain a separate analysis.

**Experiment decision:** prioritise a bounded feasibility preflight for a KITTI monocular ImVoxelNet baseline if one additional benchmark is pursued. Retain a six-camera nuScenes model as an alternative, not an additional commitment. Defer physical multi-LiDAR measurements unless lawful data access, a compatible accessible checkpoint and a measured resource budget are all established. This is a project scope recommendation; supervisor agreement has not been recorded.

## Configuration matrix

`C` counts physical camera inputs used by the system, and `L` counts physically separate LiDAR units. Dataset sensor suites can contain additional sensors that the tested pipeline does not consume. Results below are from recorded data; hardware deployment has not been demonstrated.

| Configuration | Existing project evidence | Valid interpretation | Remaining requirement |
|---|---|---|---|
| 1C, 0L | KITTI image-detector pilots | Camera 2D detection exists; the separate projected-LiDAR depth experiments also use LiDAR and are not camera-only depth evidence | A monocular 3D checkpoint evaluated on the same KITTI validation IDs and AP40 protocol |
| 2C, 0L | No stereo 3D benchmark | Two cameras could form a stereo pair or cover different directions; these are different designs | Define placement, overlap, calibration and a compatible model before assigning accuracy |
| 6C, 0L | No measured camera-only nuScenes 3D result | A camera-only surround baseline is feasible in principle | Checkpoint, framework/data adapter, temporal images if required, smoke and same-token evaluation |
| 0C, 1L | KITTI PointPillars/PV-RCNN; nuScenes LiDAR baseline | Established LiDAR-only reference pipelines | Existing evidence can be reused |
| 1C, 1L | KITTI image-candidate + PointPillars CLOCs evaluation | This combination is already measured in the project; gains are inconsistent across seeds/difficulties | Preserve the small-data and source-recovery limitations; no new nuScenes result follows |
| 2C available, 1L | nuScenes front/back retained, four views masked | Missing-view robustness of a model that still executes six view slots | Purpose-trained two-camera performance and actual compute savings are unmeasured |
| 5C available, 1L | nuScenes front camera masked | A particular camera-absence scenario | Does not establish the best five-camera placement |
| 6C, 1L | Full nuScenes validation and separate runtime study | Established complete pretrained fusion pipeline | Monetary cost and deployment feasibility require separate evidence |
| 0C, 2L or 3L | No experiment | Genuine multi-LiDAR configurations require separate sensor origins/calibrations | New data/model access and a within-dataset sensor-subset protocol |
| Multiple C, multiple L | No experiment | A further multimodal configuration, not established by the existing fusion runs | Verify both sensor-specific inputs and the model's supported fusion path |

The existing KITTI CLOCs combination should not be described as untested. Its aligned implementation uses image candidates identified in the saved records as YOLO26x, alongside PointPillars. It should not be silently relabelled YOLOv8. KITTI's acquisition vehicle had several cameras, but the relevant monocular image stream is `image_2`; physical rig inventory and model input count are distinct [S1, P1].

nuScenes supplies six cameras, one LiDAR and five radars [S2]. The evaluated BEVFusion pipelines do not use those radar streams. Nine preceding sweeps plus the keyframe are observations from one LiDAR, not ten sensors. Point thinning is a density intervention on that sensor history, not a substitute for multiple physical scanners.

## What the measured results establish

### KITTI: one LiDAR versus one camera plus one LiDAR

On the same 3,769 validation frames with Car strict IoU 0.70, PointPillars Moderate 3D AP40 is 79.1299. The three epoch-12 CLOCs seeds score 79.8616, 76.2666 and 78.7616. Only one improves Moderate; all reduce Hard AP. This is evidence against assuming that adding a camera guarantees improvement in this implementation. It does not establish a general failure of fusion. The fusion head used 96 optimisation and 32 development frames, and validation was inspected iteratively [P1].

The bounded audit found an unresolved difference between the recorded executed trainer hash and the current generic trainer. This limits faithful source reproduction; it is not an established explanation of seed variation. Keep the original scores and the inconclusive audit together [P2].

### nuScenes: one LiDAR versus six cameras plus one LiDAR

Full-validation mAP/NDS multiplied by 100 are 64.4301/69.2353 for LiDAR-only and 68.3099/71.1352 for fusion over 6,019 frames and 150 scenes. The 20 October same-evaluator replay reproduced the saved metrics. These are two separately pretrained pipelines, rather than a controlled change in camera count within one training run [P3].

In the frozen 600-frame availability study, intact fusion scores 68.1318/71.2913, compared with 63.6269/68.7638 when only front/back views are retained. LiDAR-only scores 64.3076/69.4430. The missing-view fusion model can therefore underperform the separate LiDAR baseline. Camera arrangement also matters: the class effects of front-only masking and four-view masking are not uniformly ordered by camera count [P4].

The separate single-T4, batch-one timing study reports median-of-session-medians latency of 287.8116 ms for LiDAR and 544.9786 ms for fusion, with maximum peak PyTorch allocated memory of 609.8184 and 2,231.7979 MiB. These results apply to 150 timing frames with three sessions/model. They are not timings of the masked conditions, an energy measurement or a sensor purchase cost. Accuracy context from 600 frames and full-validation accuracy from 6,019 frames must retain their separate sample scopes [P5].

## Camera-only candidate assessment

| Candidate | Relevant configuration | Documentation verified | Practical decision |
|---|---|---|---|
| ImVoxelNet, KITTI Car | One RGB camera at inference | Pinned MMDetection3D configuration uses `image_2`, `use_lidar=False`, ResNet-50 and a 1280×384 resize target. A checkpoint URL is published and returned HTTP 200 with 126,018,456 bytes in a header check | Preferred first preflight because the dataset, framework family and Car AP40 comparison already exist. Checkpoint bytes/strict load, current data access, memory and runtime remain untested |
| FCOS3D, nuScenes | One image per forward pass, six camera views in the default benchmark | Pinned configuration uses `mv_image_based`; the dataset expands cameras and the formatter explicitly merges six views with NMS | Viable route to a six-camera camera-only system. Do not report its default benchmark as a single physical camera result |
| BEVDepth R50, 256×704, two-keyframe example | Six cameras with temporal history | Official repository provides experiment code, checkpoint link and an evaluation command. The example sets `key_idxes=[-1]`; documented setup uses PyTorch 1.9.0 and MMDetection3D 1.0.0rc4 | Alternative with extra temporal-data and dependency work. Existing 600-frame current-image inventory does not prove that the required prior camera images are restored |

Sources: [S3–S8]. The published ImVoxelNet table's value labelled `mAP` is not copied into the project's strict AP40 table: metric/difficulty provenance would first need verification. The FCOS3D model-zoo memory column is also not a measured batch-one T4 requirement. A published checkpoint or an evaluation command does not prove memory fit or an acceptable running time.

A camera-only corruption comparison does **not** inherently require retraining: fixed pretrained weights can be evaluated with the same declared image interventions. It would measure inference-time sensitivity of that pipeline. Different backbones, resolution, temporal input and training histories still prevent interpreting the result as a controlled modality-only ablation. RGB-only at inference also does not necessarily mean that LiDAR-derived supervision was absent during training.

### Bounded next-experiment gate

If an additional camera-only experiment is selected, first check the KITTI path:

1. Verify the original 3,769 validation IDs, `image_2` images, calibration and labels are accessible. This assessment did not inspect the user's current Drive or Mac sensor files.
2. Retrieve the documented ImVoxelNet checkpoint, record its complete SHA-256 and provenance, and test a strict load in an isolated pinned environment. The present header check is only a reachability/size check.
3. Confirm image/calibration-only inference and correct global/camera-coordinate export on a deterministic small fixture. Labels may be used by the evaluator, never by the model at inference.
4. Measure a small pilot's actual time and peak memory before budgeting the remaining validation. No runtime estimate is asserted before that measurement.
5. Freeze the evaluation policy: Car, IoU 0.70, Easy/Moderate/Hard, common AP40 evaluator and identical validation IDs. Inspect any method-specific range/image eligibility filters; do not silently change GT to favour a configuration. Preserve full 3D predictions, including empty outputs.
6. Run the selected validation only after these checks pass. Add matched runtime measurements only if necessary for a compute comparison; timings from unrelated protocols cannot be reused as equivalent.

This would add the missing camera-only arm beside existing KITTI LiDAR and CLOCs results. It remains a comparison of complete pretrained/trained pipelines with unequal training histories. Published values cannot fill its measured-results row.

## Physical multi-LiDAR assessment

Waymo's Perception dataset contains a top LiDAR and four shorter-range units at front, side-left, side-right and rear. This provides separate calibrated sensor inputs [S9]. Use the Perception data, not assume that another Waymo product contains the same raw sensors.

OpenPCDet's inspected conversion code applies each scanner's extrinsic transform, converts points into vehicle coordinates, concatenates per-sensor points and retains counts for each LiDAR. Thus an early geometric merge can feed a point-cloud detector; an extra encoder branch and retraining are **not universal prerequisites** [S10]. Nevertheless, dropping sensors from a detector trained on a merged rig tests sensitivity under its training distribution mismatch; it does not establish the performance of an independently optimised lower-cost rig.

OpenPCDet supplies Waymo configurations and results but explicitly states that it does not distribute the corresponding pretrained models, citing the dataset agreement [S11]. This is a restriction of the inspected source, not a claim that no compatible checkpoints exist anywhere. No accessible, authorised checkpoint, local Waymo inventory, working environment or measured preparation/inference budget has been verified for this project. These concrete gaps justify deferring this experiment now, rather than asserting that multi-LiDAR fusion is technically impossible.

### Proposed follow-on protocol

- Use the same declared Waymo Perception validation segments and timestamps for top-only, top+front, top+front+rear and all-five conditions. Verify the actual scanner names and their calibrated poses from released records. Two/three-sensor choices are predefined placements, not the best arrangement discovered on validation.
- Retain sensor identity before concatenation; apply the released calibration and motion-compensation conventions. Hold return policy, time history, point features, voxel limits and post-processing fixed. Record per-sensor point counts and hashes.
- Keep GT eligibility fixed across conditions. In particular, do not recalculate difficulty or exclude objects based on only the retained sensors, which would change the task with sensor count. Preserve the official no-label-zone and difficulty rules.
- Report official Waymo AP/APH and difficulty levels, class/range breakdowns and matching token sets. Do not label these nuScenes NDS or compare their raw values directly with KITTI AP40.
- Initially evaluate a fixed compatible checkpoint as a sensor-removal sensitivity study. A deployment-optimisation comparison would additionally require training/adapting each configuration under a stated budget.
- Measure latency/memory on a common setup and obtain actual component/integration costs. Separate improvements caused by extra physical viewpoints from temporal accumulation or merely increased point count; any equal-point-budget diagnostic is a secondary intervention.

No numerical hypothesis or success threshold should be selected after inspecting these future outputs. More sensors may change coverage and redundancy; the accuracy and cost consequences remain to be measured.

## Cost connection and compact report presentation

For `n_C` cameras and `n_L` LiDARs, use a scenario expression such as `sensor cost = n_C*C_C + sum(C_L,i)`. Separate unit models when scanners differ; a Waymo top-plus-short-range rig cannot be priced as three identical LiDARs without an explicit assumption. Add interfaces, optics, synchronisation, calibration, mounting, compute, power and maintenance as separately sourced items or labelled assumptions.

The current project does not support a monetary optimum. Prices for a modern replacement product cannot inherit the measured KITTI/nuScenes accuracy. Likewise, the front/back masking experiment cannot be assigned the savings of a physical two-camera implementation while retaining measured six-camera computation.

For the 40–50-page dissertation, use one shortened configuration table in the main text and place this detailed assessment, candidate checkpoints and proposed protocols in supporting material. Existing robustness plots can remain composite figures; this assessment needs no additional image.

## Sources and evidence

All external source inspections below were performed on 21 October 2026. GitHub source versions and blob hashes are recorded in [the source register](../results/sensor_configuration/20261021/source_register.json).

- **S1:** Geiger et al., KITTI CVPR 2012 dataset paper, sensor setup: https://www.cvlibs.net/projects/autonomous_vision_survey/literature/Geiger2012CVPR.pdf
- **S2:** Caesar et al., nuScenes CVPR 2020: https://openaccess.thecvf.com/content_CVPR_2020/html/Caesar_nuScenes_A_Multimodal_Dataset_for_Autonomous_Driving_CVPR_2020_paper.html
- **S3:** Pinned MMDetection3D ImVoxelNet README and linked checkpoint: https://github.com/open-mmlab/mmdetection3d/blob/fe25f7a51d36e3702f961e198894580d83c4387b/configs/imvoxelnet/README.md
- **S4:** Pinned ImVoxelNet KITTI configuration: https://github.com/open-mmlab/mmdetection3d/blob/fe25f7a51d36e3702f961e198894580d83c4387b/configs/imvoxelnet/imvoxelnet_8xb4_kitti-3d-car.py
- **S5:** Pinned FCOS3D README: https://github.com/open-mmlab/mmdetection3d/blob/fe25f7a51d36e3702f961e198894580d83c4387b/configs/fcos3d/README.md
- **S6:** Pinned nuScenes monocular configuration, dataset expansion and formatter: https://github.com/open-mmlab/mmdetection3d/blob/fe25f7a51d36e3702f961e198894580d83c4387b/configs/_base_/datasets/nus-mono3d.py ; https://github.com/open-mmlab/mmdetection3d/blob/fe25f7a51d36e3702f961e198894580d83c4387b/mmdet3d/datasets/nuscenes_dataset.py ; https://github.com/open-mmlab/mmdetection3d/blob/fe25f7a51d36e3702f961e198894580d83c4387b/mmdet3d/evaluation/metrics/nuscenes_metric.py
- **S7:** Official BEVDepth installation, evaluation and model links: https://github.com/Megvii-BaseDetection/BEVDepth/blob/d78c7b58b10b9ada940462ba83ab24d99cae5833/README.md
- **S8:** BEVDepth temporal example: https://github.com/Megvii-BaseDetection/BEVDepth/blob/d78c7b58b10b9ada940462ba83ab24d99cae5833/bevdepth/exps/nuscenes/mv/bev_depth_lss_r50_256x704_128x128_24e_2key.py
- **S9:** Waymo Perception sensor documentation: https://waymo.com/open/data/perception/
- **S10:** OpenPCDet Waymo conversion: https://github.com/open-mmlab/OpenPCDet/blob/233f849829b6ac19afb8af8837a0246890908755/pcdet/datasets/waymo/waymo_utils.py
- **S11:** OpenPCDet Waymo model distribution note: https://github.com/open-mmlab/OpenPCDet/blob/233f849829b6ac19afb8af8837a0246890908755/README.md
- **P1:** [CLOCs aligned findings](../results/kitti/clocs/aligned_seed_study/FINDINGS.md), [three-seed diagnostics](../results/kitti/clocs/seed_diagnostics/FINDINGS.md), [project guide](PROJECT_GUIDE.md).
- **P2:** [Bounded CLOCs audit](../results/kitti/clocs/audit_20261020/README.md).
- **P3:** [Full nuScenes validation](../results/nuscenes/bevfusion/full_validation/FINDINGS.md), [full replay](../results/nuscenes/bevfusion/full_validation/replay_20261020/README.md).
- **P4:** [Camera-availability findings](../results/nuscenes/bevfusion/camera_availability/20261014/FINDINGS.md).
- **P5:** [T4 timing findings](../results/nuscenes/bevfusion/runtime/20261015/FINDINGS.md).
