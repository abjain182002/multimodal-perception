# Results and Artifact Policy

The `results/` directory stores **small, frozen artifacts that are useful for reproducibility, supervisor review, and dissertation writing**.

## Currently version-controlled

### Depth-estimation study

- `depth_method_comparison.csv`
- `depth_method_comparison.png`

These summarize the V1/V2/V3 development and independent-holdout comparison.

### Camera-detector pilot

- `camera_detector_comparison.csv`
- `camera_detector_comparison.png`

These summarize the controlled fixed-operating-point YOLOv8x vs Falcon+dedup pilot on the matched development and holdout frame sets.

## What belongs here

Commit:

- compact CSV/JSON summary tables,
- final comparison figures,
- selected qualitative examples that support a specific finding,
- important failure-case visualizations,
- small artifacts explicitly referenced in the dissertation or supervisor updates.

Do **not** commit:

- the KITTI or nuScenes datasets,
- virtual environments,
- model weights,
- hundreds or thousands of generated frames,
- duplicate screenshots,
- temporary debugging images,
- bulk contents of `model_outputs/`.

Bulk generated outputs stay local and are intentionally excluded by `.gitignore`.

## Planned qualitative examples

When a representative image is worth preserving, use a structure such as:

```text
results/qualitative/
├── yolov8_lidar_000149.png
├── falcon_lidar_000149.png
└── falcon_duplicate_failure_006648.png
```

Only selected, verified images should be added. Their purpose should be documented in `PROJECT_HISTORY.md` or `DAILY_LOG.md`.
