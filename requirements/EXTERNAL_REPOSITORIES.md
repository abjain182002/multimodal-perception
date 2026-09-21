# External Repository Provenance

This project uses established open-source implementations as external references and reproducible baselines. Third-party repositories are kept outside this repository under `~/Major_Project/external/`.

The external implementations are used as baselines or supporting frameworks. Project-specific work is maintained in this repository, including integration code, evaluation scripts, experimental results, comparative analysis and visualisation.

## MMDetection3D

- Repository: `open-mmlab/mmdetection3d`
- Local directory: `~/Major_Project/external/mmdetection3d`
- Commit: `fe25f7a5`
- Branch: `main`
- Licence: Apache License 2.0
- Intended use:
  - 3D detection framework
  - PointPillars investigation
  - nuScenes dataset support
  - 3D evaluation utilities
  - possible advanced-fusion support

## OpenPCDet

- Repository: `open-mmlab/OpenPCDet`
- Local directory: `~/Major_Project/external/OpenPCDet`
- Commit: `233f849`
- Branch: `master`
- Licence: Apache License 2.0
- Intended use:
  - LiDAR-only 3D detection
  - PointPillars
  - PV-RCNN++ reference

## BEVFusion

- Repository: `ADLab-AutoDrive/BEVFusion`
- Local directory: `~/Major_Project/external/BEVFusion`
- Commit: `fb4384c`
- Branch: `main`
- Licence: Apache License 2.0
- Intended use:
  - advanced camera-LiDAR fusion reference
  - possible nuScenes multimodal experiment

## MultiCorrupt

- Repository: `ika-rwth-aachen/MultiCorrupt`
- Local directory: `~/Major_Project/external/MultiCorrupt`
- Commit: `39652e3`
- Branch: `main`
- Licence: MIT
- Intended use:
  - multimodal robustness experiments
  - corruption methodology and implementation reference

## Falcon Perception

- Repository: `tiiuae/Falcon-Perception`
- Local directory: `~/Major_Project/external/Falcon-Perception`
- Commit: `59a845a`
- Licence: Apache License 2.0
- Project use:
  - open-vocabulary camera perception
  - KITTI camera evaluation
  - Falcon + LiDAR controlled fusion experiments

## Lidar-CameraFusion-3DobjectDetection

- Repository: `Azitt/Lidar-CameraFusion-3DobjectDetection`
- Local directory: `~/Major_Project/external/Lidar-CameraFusion-3DobjectDetection`
- Commit: `b59b49d`
- Licence: no repository licence file confirmed during provenance review
- Project use:
  - early reference for KITTI camera-LiDAR fusion
  - YOLOv8 + PV-RCNN++ architecture reference
  - early-vs-late fusion methodology reference

Because a repository licence was not confirmed, its source code should not be redistributed as project-owned code without further licence verification.

## Use of third-party implementations

Third-party implementations are treated as reproduced or adapted baselines and are not presented as original project implementations.

For every experiment using third-party software, the project record should identify:

- source repository,
- paper/authors where applicable,
- exact commit or version,
- pretrained weights,
- dataset and configuration,
- reused components,
- project-specific modifications,
- generated results,
- project-specific evaluation and analysis,
- limitations.

Published metrics from external repositories or papers must not be presented as experimentally obtained project results unless they were independently reproduced.
