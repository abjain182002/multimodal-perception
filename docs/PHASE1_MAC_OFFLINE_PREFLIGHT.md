# Phase 1 Mac preflight (27 September work prepared on 26 September)

This records executable paths and exact missing inputs while Colab GPU access
is unavailable. It does **not** claim new model inference, nuScenes coverage
or trained fusion. Repository revisions below are the project pins in
`requirements/EXTERNAL_REPOSITORIES.md`; the Mac inventory must independently
verify its own checkouts.

## One Mac inventory

From `azitt310` at the repository root:

```bash
python -m scripts.audit_phase1_mac_dependencies \
  --root "$HOME/Major_Project" \
  --manifest results/kitti/clocs/training_exports/clocs_train_export_batch_manifest.json \
  --output results/audits/environment/phase1_mac_dependency_inventory.json
```

The JSON lists actual checkout SHAs (when a checkout exists), relevant config
and weight paths, the pilot's uncompressed input bytes, disk free bytes, and
candidate nuScenes metadata directories. `present: true` is **not** proof
that an SDK can import, a checkpoint matches the code, nuScenes sensor files
are complete, or a model can run. Review the JSON before committing it.

## Distinct baselines and concrete execution gates

| Path | Verified source code at pinned revision | Required before an experimental result |
|---|---|---|
| Current CLOCs/PointPillars | Our [single-frame exporter](../scripts/export_clocs_dense_train_frame.py) plus [frozen split](../configs/splits/kitti_second_split.json) | Restore and verify the exact pilot archive; CUDA checkpoint/config pin, dense pre-NMS exports, target checks and later training. Pinned native evaluator remains a separate open crash. |
| Azitt early/late fusion | [Pinned main.py](https://github.com/Azitt/Lidar-CameraFusion-3DobjectDetection/blob/b59b49d2e915cd3b4a3b816b9c228e811601e241/main.py) and [FramePredictor](https://github.com/Azitt/Lidar-CameraFusion-3DobjectDetection/blob/b59b49d2e915cd3b4a3b816b9c228e811601e241/inference/predictor.py) | `main.py` hardcodes KITTI `000149`, a dataset index, `model.cuda()` and `lidar-camera-models/pv_rcnn_8369.pth`; `predict_single_frame` calls the early and late methods. Audit the exact weights, actual frame/split IDs and a metric-producing loop. Its illustrative GIFs are not our measured AP. `spconv-cu114` in [requirements](https://github.com/Azitt/Lidar-CameraFusion-3DobjectDetection/blob/b59b49d2e915cd3b4a3b816b9c228e811601e241/requirements.txt) makes this a CUDA-specific environment. |
| OpenPCDet PV-RCNN++ | [Pinned KITTI community config](https://github.com/open-mmlab/OpenPCDet/blob/233f849829b6ac19afb8af8837a0246890908755/tools/cfgs/kitti_models/pv_rcnn_plusplus_reproduced_by_community.yaml) and [install guide](https://github.com/open-mmlab/OpenPCDet/blob/233f849829b6ac19afb8af8837a0246890908755/docs/INSTALL.md) | The config is explicitly community-reproduced. Establish a compatible Linux/PyTorch/CUDA/`spconv` build, actual compatible checkpoint, KITTI preparation, predictions and strict evaluation; label it as community config when comparing. |
| ADLab BEVFusion | [Pinned nuScenes multimodal config](https://github.com/ADLab-AutoDrive/BEVFusion/blob/fb4384ca05cae05be548dd96a0f940da70263cdb/configs/bevfusion/bevf_pp_2x8_1x_nusc.py), [dataset preparation](https://github.com/ADLab-AutoDrive/BEVFusion/blob/fb4384ca05cae05be548dd96a0f940da70263cdb/docs/data_preparation.md) and [install guide](https://github.com/ADLab-AutoDrive/BEVFusion/blob/fb4384ca05cae05be548dd96a0f940da70263cdb/docs/getting_started.md) | Confirm a matching config/checkpoint, nuScenes data and generated info files, and a separately compatible CUDA environment. The pinned guide uses legacy `mmdet`/`mmcv-full`; do not assume compatibility with our MMDetection3D 1.4 PointPillars environment. |
| PointPainting-style semantic fusion | [Original paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Vora_PointPainting_Sequential_Fusion_for_3D_Object_Detection_CVPR_2020_paper.pdf) | Define a semantic segmentation model and per-point projected class scores, detector feature-input modification and a controlled retraining/evaluation protocol. Existing YOLO boxes alone are **not** PointPainting. |

Missing model weights, SDK compatibility, sensor data, or CUDA access are
explicit experimental blockers. Keep these five model paths separate from
the already measured projected-LiDAR *depth* pilot.

## Second dataset protocol: nuScenes

- The [official schema](https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuscenes.md)
  links samples to `sample_data` by tokens, and each sensor record to a
  `calibrated_sensor` and time-specific `ego_pose`. A KITTI
  `P2/R0_rect/Tr_velo_to_cam` projection cannot simply be reused. Check
  the camera intrinsics and sensor-to-ego, ego-to-global, and timestamp
  relationships before projecting LiDAR into each camera.
- Start with the explicit `KITTI Car -> nuScenes car` class mapping; verify
  exact source categories against the
  [official detection taxonomy](https://www.nuscenes.org/object-detection).
  Avoid silently relabelling vans, trailers and trucks as `car`.
- Use the [official nuScenes detection configuration](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/configs/detection_cvpr_2019.json)
  with its class ranges and center-distance thresholds. Keep nuScenes
  mAP/NDS and KITTI IoU-based 3D AP40 in separate columns. A mini-set is a
  feasibility check, not full nuScenes validation.
- A metadata folder on disk does not establish complete `samples/`,
  `sweeps/` or matching pretrained checkpoints. The pinned
  [BEVFusion preparation guide](https://github.com/ADLab-AutoDrive/BEVFusion/blob/fb4384ca05cae05be548dd96a0f940da70263cdb/docs/data_preparation.md)
  specifies the required layout. Record the exact version, available sensor
  modalities and frame counts once verified on the Mac/Colab.

## Next evidence gate

Inspect the Mac dependency JSON and the SHA-verified 16-frame archive. Save
small reviewed manifests in GitHub; keep the raw KITTI archive outside Git.
On the next available CUDA runtime, verify the archive first, then process
the fixed train-only queue. No validation-frame hyperparameter decisions.

