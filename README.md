# Multimodal Perception

MSc Major Project: Software-only deep learning pipeline for multi-modal object detection and robotic perception using camera and LiDAR fusion.

## Project Status

**Technical implementation: approximately 51% complete** (working estimate, 1 September 2026).

Latest verified milestone: **Falcon Perception + KITTI LiDAR single-frame fusion** using the frozen projected-LiDAR median-depth association method.

Next technical milestone: multi-frame Falcon + LiDAR evaluation, followed by a controlled YOLO+LiDAR vs Falcon+LiDAR comparison.

## Project Records

- [`PROJECT_HISTORY.md`](PROJECT_HISTORY.md) — master technical/dissertation record: project rationale, experiments, numerical results, failures, fixes, design decisions, limitations, and dissertation-safe interpretations.
- [`DAILY_LOG.md`](DAILY_LOG.md) — chronological work-session record for new experiments and engineering changes.
- [`requirements/ENVIRONMENTS.md`](requirements/ENVIRONMENTS.md) — environment strategy and reproducibility notes.

## Frozen Results

- [`results/depth_method_comparison.csv`](results/depth_method_comparison.csv)
- [`results/depth_method_comparison.png`](results/depth_method_comparison.png)
- [`results/camera_detector_comparison.csv`](results/camera_detector_comparison.csv)
- [`results/camera_detector_comparison.png`](results/camera_detector_comparison.png)

See [`results/README.md`](results/README.md) for the artifact-storage policy.

## Storage Policy

Datasets, virtual environments, model weights, and bulk generated `model_outputs/` are intentionally excluded from Git history. Source code, evaluation scripts, compact numerical results, dissertation-ready figures, and selected qualitative examples are version-controlled.
