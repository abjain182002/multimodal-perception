# Train-only CLOCs export pilot

The next GPU batch is selected **before** GPU inference from the frozen SECOND
training split. Selection uses existing KITTI Car ground-truth counts for
input coverage, not model predictions. The pilot deliberately includes empty,
low-count, medium-count and crowded frames; it must not be reported as a
representative training corpus or a final model evaluation.

## Mac input-audit gate

Run in the repository root using the environment with NumPy and KITTI
calibration helpers installed (`azitt310`):

```bash
python -m scripts.prepare_clocs_train_export_batch \
  --dataset "$HOME/Major_Project/datasets/KITTI/training" \
  --train "$HOME/Major_Project/external/CLOCs/second/data/ImageSets/train.txt" \
  --val "$HOME/Major_Project/external/CLOCs/second/data/ImageSets/val.txt" \
  --size 16 --seed 20260926 --include 000000 000149 \
  --output results/clocs_train_export_batch_manifest.json
```

The command verifies the exact frozen split, reads range-filtered Car labels
on every training frame, and hashes the four input files for each selected
frame. It fails for missing inputs or validation-frame inclusion. Inspect the
printed selected IDs and counts, then commit the small JSON manifest. There
are **no new 3D candidate exports** merely because this gate passes.

The manifest's `frames` array is the ordered GPU work queue. Restore those
exact KITTI image, raw Velodyne, label and calibration files on Colab and
confirm their SHA-256 values against the manifest before running the pinned
single-frame export script. Use the already frozen checkpoint, config and
split in `scripts.export_clocs_dense_train_frame`; preserve each candidate
NPZ and its SHA-256 outside Git. Frame `000000` is the known empty-GT
control and `000149` is the known positive-GT control. A pilot of sixteen
frames alone is insufficient to claim trained fusion accuracy.
