# PV-RCNN full KITTI validation — 30 September 2026

The official pretrained OpenPCDet PV-RCNN checkpoint scored 92.1047 / 84.3604 / 82.4830 Car 3D AP40 (Easy / Moderate / Hard) on the frozen 3,769-frame SECOND validation split at strict IoU 0.70. Native OpenPCDet and the pinned MMDetection3D evaluator agreed exactly across all nine 2D/BEV/3D metrics.

Relative to the verified PointPillars baseline, PV-RCNN gained 1.7598 / 5.2306 / 6.3265 3D AP points. This establishes a stronger LiDAR-only baseline in this experiment. It does not establish an advantage from fusion, equal training budgets, statistical significance or universal model superiority. Both detectors are pretrained; the CLOCs fusion head used 96 optimization and 32 development frames. Checkpoint-specific upstream training provenance should accompany the comparison.

The model is PV-RCNN, not PV-RCNN++ or Azitt. No PV-RCNN camera fusion, retraining or cross-dataset evaluation has yet been measured. A separate synchronized batch-one T4 runtime benchmark is now recorded in [runtime findings](../runtime/FINDINGS.md): 96 calls after 10 warm-ups, pipeline median 176.31 ms and p95 196.16 ms. It is not a cross-model runtime comparison.

The implementation reused OpenPCDet commit 233f849829b6ac19afb8af8837a0246890908755 and its official KITTI model-zoo checkpoint. All 367/367 checkpoint parameters loaded. Camera-FOV filtering, voxelization and KITTI prediction conversion used upstream code. The local wrapper provides headless/resumable inference, integrity checks and cross-framework evaluator verification.

Setup used Python 3.10, Torch 2.1.2+cu121, NumPy 1.26.4 and spconv-cu120 2.3.6 on a Tesla T4. The installed wheel omitted Python helper folders; missing files were restored from the same pinned checkout. Kornia 0.6.7 supplies the API required by the Argoverse helper imported by OpenPCDet. See environment.txt for resolved packages.

Figure caption: Car strict IoU 0.70 AP40 on the same 3,769 KITTI validation frames with a common evaluator. PointPillars and CLOCs use the verified reduced-FOV run; PV-RCNN uses official OpenPCDet preprocessing. Architecture, training recipes, point-cloud ranges and postprocessing differ. These are complete-pipeline comparisons, not an isolated architecture ablation.

Sources: [OpenPCDet](https://github.com/open-mmlab/OpenPCDet/tree/233f849829b6ac19afb8af8837a0246890908755), [PV-RCNN paper, Shi et al., CVPR 2020](https://arxiv.org/abs/1912.13192). Reused code and pretrained weights must be attributed; these measurements are our reproduction, not copied model-zoo scores.

