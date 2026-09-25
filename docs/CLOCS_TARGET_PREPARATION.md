# CLOCs target preparation: frozen KITTI Car protocol

## Verified prerequisites

- Frozen SECOND split: `configs/splits/kitti_second_split.json` and the original CLOCs `train.txt` / `val.txt`; 3,712 / 3,769 frames and zero overlap.
- Complete GT audit: `results/kitti_clocs_gt_audit.json`; 28,374 range-filtered Car boxes over 7,481 frames.
- Dense candidate arrays must originate from the *same* frozen PointPillars extraction as their corresponding 2D features. The saved structural smoke output has no real 3D candidate boxes and is unsuitable for training.

## GPU export contract (one KITTI frame)

Run dense PointPillars extraction on the available CUDA runtime. Save these arrays **in the original pre-NMS order** from `extract_dense_pointpillars_candidates`:

```python
import numpy as np

from fusion.clocs import build_clocs_feature_rows

frame_id = "000001"  # example; use the actual six-digit KITTI frame ID
# candidates = extract_dense_pointpillars_candidates(model, pointcloud_path,
#     calibration_path, image_width, image_height)
# boxes_2d and scores_2d are detections from that same frame.
required = ("boxes_camera", "boxes_lidar", "scores_3d", "distance_82",
            "candidate_index", "projected_boxes")
assert all(key in candidates for key in required)
np.savez_compressed(
    "000001_dense_candidates.npz", frame_id=frame_id,
    **{key: candidates[key] for key in required},
)
rows = build_clocs_feature_rows(
    candidates["projected_boxes"], candidates["scores_3d"],
    boxes_2d, scores_2d, candidates["distance_82"],
)
np.savez_compressed(
    "000001_clocs_rows.npz", frame_id=frame_id,
    boxes_2d=boxes_2d, scores_2d=scores_2d,
    features=rows["features"], candidate_index=rows["candidate_index"],
    detector_2d_index=rows["detector_2d_index"], row_type=rows["row_type"],
)
```

Use frame-specific names instead of the example literals for subsequent frames. Keep dataset files, large candidate archives, checkpoints, and 2D/3D weights outside Git; record detector/checkpoint hashes, extraction settings, split hash and both NPZ hashes in the experiment manifest.

## CPU conversion on the Mac

```bash
cd ~/Major_Project/multimodal-perception
python -m scripts.build_clocs_training_example \
  --dataset "$HOME/Major_Project/datasets/KITTI/training" \
  --frame-id 000001 \
  --candidates "$HOME/Major_Project/model_outputs/clocs_stage5_checkpoints/000001_dense_candidates.npz" \
  --features "$HOME/Major_Project/model_outputs/clocs_stage5_checkpoints/000001_clocs_rows.npz" \
  --train "$HOME/Major_Project/external/CLOCs/second/data/ImageSets/train.txt" \
  --val "$HOME/Major_Project/external/CLOCs/second/data/ImageSets/val.txt" \
  --output "$HOME/Major_Project/model_outputs/clocs_stage5_checkpoints/000001_real_targets.npz"
```

The command checks split identity and frame IDs, dense candidate ordering, feature-score and feature-distance alignment, and **rebuilds all feature rows** from the saved projected boxes and original 2D detections. It checks the candidate and 2D indices, row types, IoUs, and scores before GT range filtering and computing camera-coordinate rotated **3D** IoU: positive `>=0.70`, negative `<=0.50`, and ignored in between. It verifies the saved example's SHA-256. Frames without retained Car GT legitimately have all-negative candidates. An isolated all-negative frame is not proof of a dataset-level training signal.

Before training, cross-check sampled frame IoUs against the pinned SECOND/CLOCs evaluation kernel, inspect positive/negative/ignored counts across multiple train frames, verify example hashes and frame membership, then freeze validation evaluation settings. The CPU reference implementation and synthetic tests are preparation, not an AP result.
