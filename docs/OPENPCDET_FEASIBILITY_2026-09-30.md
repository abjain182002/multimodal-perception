# OpenPCDet next experiment: feasibility gate

Reviewed official OpenPCDet source on 30 September 2026, commit `233f849829b6ac19afb8af8837a0246890908755`.

## Decision

Proceed first with a KITTI **PV-RCNN LiDAR-only reproduction**, contingent on checkpoint download and compiled CUDA operations passing. It provides a stronger detector comparison using the existing KITTI data. It is not Azitt, PV-RCNN++, semantic fusion, or a replacement for the verified CLOCs result.

The official KITTI model zoo lists PV-RCNN with a downloadable checkpoint. Its published table uses R11; our experiment must recompute AP40 on the frozen split rather than copying that number. The reviewed table does not establish a downloadable KITTI PV-RCNN++ checkpoint. Azitt remains blocked pending the correct identifiable weights.

## Official sources

- [Pinned model zoo](https://github.com/open-mmlab/OpenPCDet/blob/233f849829b6ac19afb8af8837a0246890908755/README.md)
- [Installation](https://github.com/open-mmlab/OpenPCDet/blob/233f849829b6ac19afb8af8837a0246890908755/docs/INSTALL.md)
- [PV-RCNN KITTI configuration](https://github.com/open-mmlab/OpenPCDet/blob/233f849829b6ac19afb8af8837a0246890908755/tools/cfgs/kitti_models/pv_rcnn.yaml)
- [Official checkpoint link](https://drive.google.com/file/d/1lIOq4Hxr0W3qsX83ilQv0nk1Cls6KAr-/view)
- [Demo instructions](https://github.com/open-mmlab/OpenPCDet/blob/233f849829b6ac19afb8af8837a0246890908755/docs/DEMO.md)

## Execution gates

1. Download the linked weights and record filename, byte count, SHA-256, source URL and date. Download availability and checkpoint compatibility have not yet been verified.
2. Create a separate Linux GPU environment. OpenPCDet uses compiled CUDA operations and spconv; the official installation document lists older tested Torch versions, so compatibility with the available Kaggle toolchain must be demonstrated. Preserve the CLOCs environment.
3. Pin the source commit above and record Python/Torch/CUDA/spconv/compiler versions. Verify CUDA operations before detector inference.
4. Run headless inference on four fixed KITTI frames using the official configuration and frozen preprocessing. Save boxes, scores, labels and hashes. A GUI demo is unsuitable for Kaggle.
5. If checkpoint loading and finite-box smoke checks pass, prepare KITTI validation metadata, confirm the same ordered 3,769 IDs and evaluate strict Car AP40 using a recorded evaluator. Never interpret a four-frame smoke test as full accuracy.
6. Only after a measured baseline should detector outputs be adapted to our fusion interface. Dense proposal geometry and calibration must be re-audited; PointPillars anchors/checkpoints cannot be reused directly.

## Scope and exit criteria

Today the source/checkpoint route is identified; no PV-RCNN inference or AP result has been produced. If weights or CUDA build compatibility block progress, record the blocker rather than claiming an integration. BEVFusion/nuScenes and MultiCorrupt need their own dataset/checkpoint gates and remain unmeasured. NVIDIA optimization stays future work.

Use and cite OpenPCDet and the PV-RCNN paper (Shi et al., CVPR 2020, [arXiv:1912.13192](https://arxiv.org/abs/1912.13192)); report this as pretrained-model reproduction plus our adapters/evaluation. Do not copy upstream results as locally measured evidence.
