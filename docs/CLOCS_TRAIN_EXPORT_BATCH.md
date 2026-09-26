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

## Offline transfer while the GPU is unavailable

The trusted manifest already records hashes of all **64** selected source
files and the two canonical split files. Create one checked archive on the
Mac (the archive itself stays outside Git):

```bash
python -m scripts.package_clocs_train_export_batch pack \
  --manifest results/clocs_train_export_batch_manifest.json \
  --dataset "$HOME/Major_Project/datasets/KITTI/training" \
  --train "$HOME/Major_Project/external/CLOCs/second/data/ImageSets/train.txt" \
  --val "$HOME/Major_Project/external/CLOCs/second/data/ImageSets/val.txt" \
  --output "$HOME/Desktop/clocs_train_export_16.tar.gz"
```

The tool checks every source and split SHA-256 before packing, embeds the
exact Git manifest, verifies all archive members after packing, and prints
the archive SHA-256 and bytes. Keep that output for the transfer record; do
not commit the large archive. A Colab CPU runtime can receive it when
convenient, even if a GPU runtime is unavailable.

After transferring the exact archive *and* pulling the repository containing
the manifest on Colab, verify and restore it with Python (no GPU needed):

```bash
python -m scripts.package_clocs_train_export_batch verify \
  --manifest results/clocs_train_export_batch_manifest.json \
  --archive /content/clocs_train_export_16.tar.gz \
  --extract-to /content/datasets/KITTI
```

The resulting paths are under `/content/datasets/KITTI/training/` for the four
KITTI modalities and `/content/datasets/KITTI/splits/{train,val}.txt` for
the pinned splits. The verifier rejects missing, extra, duplicated,
non-regular and modified archive members before extraction. Compare the
printed archive SHA-256 with the Mac run. Transfer is an input milestone only:
the CUDA model and pinned evaluator still require a compatible GPU runtime.
