# Results overview

Results are physically grouped by dataset, method and experiment. Open an experiment's README to find its figures, tables and evidence.

## KITTI

| Study | Folder | Status |
|---|---|---|
| Camera detector comparison | [Camera](kitti/camera/detector_comparison/) | Fixed-operating-point pilot |
| Camera detections with LiDAR depth | [Camera + depth](kitti/camera/lidar_depth_comparison/) | Conditional matched-detection depth results |
| Depth estimator V1/V2/V3 | [Depth methods](kitti/depth/method_comparison/) | Development / holdout comparison |
| PointPillars | [Baseline and cases](kitti/pointpillars/) | Full 3,769-frame validation |
| PV-RCNN | [Validation](kitti/pv_rcnn/validation/), [runtime](kitti/pv_rcnn/runtime/) | Full validation and serial runtime benchmark |
| CLOCs | [Study index](kitti/clocs/README.md) | Final results separated from pilots and historical diagnostics |

## nuScenes

| Study | Folder | Status |
|---|---|---|
| BEVFusion mini | [Mini](nuscenes/bevfusion/mini/) | 81-sample mini_val pilot |
| BEVFusion frozen validation subset | [Subset](nuscenes/bevfusion/validation_subset/) | 1,231 samples / 31 scenes; original evidence imported, replay pending |

## Shared records

- [Validated results](VALIDATED_RESULTS.md)
- [Figure gallery and case studies](../docs/FIGURE_GALLERY.md)
- [Split audits](audits/kitti/) and [environment inventory](audits/environment/)
- [Old-to-new path map](../docs/RESULTS_PATH_MIGRATION.json)

## Artifact policy

Commit compact metrics, selected figures and cases, source/protocol records and verification audits. Keep datasets, weights, environments, bulk predictions and generated frame collections outside Git. Archived manifests retain original bytes and historical paths; the migration map records current locations. File moves do not create new experimental evidence.
