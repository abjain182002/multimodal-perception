# Scripts guide

Run commands from the repository root with the appropriate environment. No model training or inference was rerun by the folder migration.

| Task | Entry points |
|---|---|
| KITTI split / GT checks | `validate_kitti_split.py`, `audit_kitti_gt.py` |
| Camera / depth figures | `generate_camera_comparison_results.py`, `generate_depth_results.py`, `generate_yolov8_yolo26_lidar_comparison.py` |
| CLOCs data preparation | `prepare_clocs_train_export_batch.py`, `run_clocs_dense_export_manifest.py`, `build_clocs_yolo26_pilot_batch.py` |
| CLOCs training | `train_clocs_pilot.py`, `train_clocs_128.py` |
| CLOCs validation | `run_clocs_full_validation.py`, `evaluate_clocs_full_validation.py` |
| CLOCs figures | `plot_clocs_aligned_results.py`, `plot_clocs_seed_diagnostics.py`, `plot_clocs_reduced_results.py` |
| PV-RCNN records / figures | `record_pv_rcnn_validation.py`, `plot_pv_rcnn_results.py`, `record_pv_rcnn_runtime.py` |
| BEVFusion records / figures | `record_bevfusion_mini.py`, `plot_bevfusion_subset_results.py` |

See [technical guide](../docs/PROJECT_GUIDE.md), [environment notes](../requirements/ENVIRONMENTS.md) and [results map](../results/README.md). Original Colab scripts under experiment evidence folders retain their historical `/content/` paths. They are archived source records, not a portable one-command setup.
