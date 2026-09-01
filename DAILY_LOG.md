# Daily Project Log

This file records work chronologically. `PROJECT_HISTORY.md` remains the synthesized master dissertation/technical record; this log preserves session-level details so that nothing important is lost between larger consolidations.

---

## 2026-09-01 — Falcon + LiDAR single-frame fusion

### Objective
Integrate Falcon Perception with the existing KITTI LiDAR projection and the already frozen median-ROI depth association method on one frame before expanding to a multi-frame experiment.

### Controlled variables
The LiDAR side was intentionally kept unchanged from the YOLO baseline:

- KITTI calibration matrices (`P2`, `R0_rect`, `Tr_velo_to_cam`),
- projected LiDAR points in camera coordinates,
- points associated when their image projection lies inside a 2D detector box,
- minimum of 3 projected LiDAR returns,
- depth estimate = median camera-coordinate LiDAR depth inside the box.

### Changed variable
The camera detector input was changed from YOLOv8x to **Falcon Perception + score-free deduplication at IoU 0.50**.

### Dataset / frame
KITTI training frame:

```text
000149
```

### Environment

```text
~/Major_Project/envs/falcon310
Python 3.10.21
```

### Implementation
Created:

```text
scripts/fuse_falcon_lidar.py
```

The script reuses `adapters/falcon_adapter.py`, converts Falcon detections to project box format, applies the already validated score-free deduplication rule, projects KITTI LiDAR into the camera frame, associates points with each deduplicated box, and estimates median depth.

### Problem encountered
The first runtime attempt failed with:

```text
ModuleNotFoundError: No module named 'matplotlib'
```

### Diagnosis
`matplotlib` had not yet been installed in the dedicated `falcon310` environment. The script itself had already passed `py_compile`, so this was an environment dependency issue rather than a Python syntax problem.

### Fix
Installed Matplotlib into `falcon310`:

```text
matplotlib 3.10.9
```

Import verification succeeded, after which the fusion script ran end-to-end.

### Result
Falcon output on frame `000149`:

```text
Raw Falcon boxes: 11
Deduplicated boxes: 10
Fused detections: 10
```

All 10 deduplicated boxes had at least 3 associated projected LiDAR points.

Median LiDAR depth estimates:

| Detection | Estimated depth | LiDAR points |
|---:|---:|---:|
| 0 | 59.97 m | 7 |
| 1 | 49.60 m | 21 |
| 2 | 41.58 m | 31 |
| 3 | 47.37 m | 66 |
| 4 | 47.44 m | 102 |
| 5 | 27.71 m | 92 |
| 6 | 27.85 m | 129 |
| 7 | 15.73 m | 309 |
| 8 | 13.21 m | 767 |
| 9 | 9.22 m | 1,629 |

### Qualitative verification
Generated locally:

```text
~/Major_Project/model_outputs/falcon_lidar_fusion/falcon_lidar_000149.png
```

The image was visually inspected. Projected LiDAR returns were aligned with the detected vehicle regions, and the near-to-far ordering of the estimated depths was physically plausible.

A presentation limitation was noted: dense LiDAR points and overlapping labels make the current visualization cluttered. This is a visualization issue, not evidence of a fusion failure, and can be cleaned later for dissertation presentation.

### Git synchronization issue
The first push of the new local commit was rejected because the remote `main` branch already contained newer documentation commits (`PROJECT_HISTORY.md` and the README link).

The repository was synchronized safely with:

```text
git pull --rebase origin main
git push
```

No force-push was used. The rebase succeeded without conflict and the final working tree was clean.

### Final commit

```text
bf3154d — Add Falcon LiDAR single-frame fusion
```

### Dissertation-safe interpretation
This experiment demonstrates successful **single-frame software integration** of Falcon detections with the frozen KITTI projected-LiDAR median-depth association pipeline. It is a qualitative/engineering validation step, not yet a multi-frame accuracy result and not evidence that Falcon+LiDAR outperforms YOLO+LiDAR.

### Next step

- multi-frame Falcon + LiDAR evaluation,
- then controlled YOLO+LiDAR vs Falcon+LiDAR comparison using matched evaluation conditions.

### Progress checkpoint
Working technical-completion estimate after this milestone: **approximately 51%**.
