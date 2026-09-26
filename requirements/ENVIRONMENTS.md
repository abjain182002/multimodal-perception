# Environment and Reproducibility Notes

The project intentionally uses **separate environments** because the local development machine is Apple Silicon while several planned 3D perception frameworks depend on Linux/NVIDIA CUDA.

## Local Apple Silicon development

### `azitt310`

Primary use:

- Python 3.10,
- Ultralytics / YOLOv8x,
- KITTI camera processing,
- LiDAR-to-camera projection,
- YOLO camera evaluation,
- projected-LiDAR fusion,
- depth analysis,
- plotting.

Known project settings used so far include:

```text
YOLO weights: yolov8x.pt
YOLO confidence threshold: 0.25
YOLO device on Mac: MPS
```

The model weight file itself is intentionally excluded from Git.

For the independent CPU check of CLOCs camera 3D IoU on Mac, `azitt310` needs the additional dependency in `requirements/macos-clocs-geometry.txt`. Install it using the active Python environment:

```bash
python -m pip install --only-binary=:all: -r requirements/macos-clocs-geometry.txt
```

This requirement pins the intended version for the geometry check; confirm and record the versions actually installed and the audit result after running it. The separate pinned CUDA evaluator comparison remains unresolved.

### `falcon310`

Primary use:

- Python 3.10.21,
- Falcon Perception,
- MLX backend on Apple Silicon,
- Falcon adapter and evaluation,
- Falcon + projected-LiDAR fusion.

Verified / relevant details from completed work:

```text
Falcon backend: MLX
Falcon dtype: float16
Falcon detection query: car
Falcon score-free dedup IoU: 0.50
Matplotlib: 3.10.9 (installed 2026-09-01 after runtime dependency error)
```

Falcon model files are fetched/cached separately and are not version-controlled in this repository.

## CUDA-heavy future stage

The following planned components should be treated as a separate Linux/NVIDIA environment rather than forced onto the Mac:

- OpenPCDet,
- PV-RCNN++,
- PointPillars configurations requiring `spconv`,
- BEVFusion,
- other CUDA-specific 3D perception code.

A reference baseline repository used earlier includes CUDA-specific dependencies such as `spconv-cu114` and hard-coded `.cuda()` calls, which is why the full reference stack has not been reproduced unchanged on Apple Silicon.

## Reproducibility rule

Do not invent or retroactively guess exact package versions that were not recorded at execution time. As the project continues, record exact versions for important environments when they are verified. Before final submission, reproducibility cleanup should include environment snapshots or requirement files generated from the verified working environments.

## Storage exclusions

The repository `.gitignore` intentionally excludes:

- virtual environments,
- datasets,
- model weights (`.pt`, `.pth`, `.ckpt`, `.onnx`),
- `outputs/` and `model_outputs/`,
- logs and editor-specific files.

This keeps Git focused on code, controlled result artifacts, and project documentation rather than large reproducible inputs that should be obtained separately.
