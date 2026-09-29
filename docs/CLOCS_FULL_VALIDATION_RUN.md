# Full KITTI CLOCs validation (29 September 2026)

## Frozen inputs

- 3,769 ordered KITTI SECOND validation frames, zero overlap with 3,712 train IDs.
- Archive `clocs_kitti_validation_3769.tar.gz`: 7,126,033,913 bytes,
  SHA-256 `376d286940943f5645f54380a3dc694cb0a0ce669bf6c9b259709a0e0e9d057c`.
- PointPillars, YOLO26x and the ratio-0 128-frame CLOCs checkpoint are SHA-checked
  in `scripts/run_clocs_full_validation.py`. Ultralytics 8.4.132 is required.
- The four-frame engineering subset, including exact GPU/simulator AP40 parity,
  does **not** estimate full KITTI AP.

## Procedure

1. Attach the unpacked validation archive and a current `git archive` code bundle
   as Kaggle datasets. Keep the pinned checkpoints and YOLO26x dataset attached.
2. Confirm the archive SHA before upload, the unpacked `manifest.json`, and
   complete `training/{velodyne,image_2,calib,label_2}` directories afterward.
3. Run `python -m scripts.run_clocs_full_validation` with `--max-new-frames 4`
   for a quick plumbing test; then rerun without the limit. Each frame is an
   independent compact JSON file. An existing frame is validated and reused.
4. Preserve the per-frame output directory after inference. A new Kaggle session
   has a new `/kaggle/working` directory; restore the outputs before resuming.
5. Run `python -m scripts.evaluate_clocs_full_validation` with
   `NUMBA_CUDA_USE_NVIDIA_BINDING=1`, `cuda-python==12.2.1` and no simulator.
   It requires all 3,769 frames and computes Car 2D/BEV/3D AP40 at IoU 0.70.
6. The evaluator compares all nine PointPillars strict AP40 numbers against the
   previously audited baseline. It refuses to report a fusion AP if the largest
   absolute discrepancy exceeds 0.05 AP points. Investigate any discrepancy
   before accepting the fusion comparison.

## Interpretation

The validation set is reserved for final comparison. The CLOCs checkpoint was
selected on the separate 128-frame train/internal-dev pilot. Do not tune model,
checkpoint, or score mixing on the validation AP. Report the complete metrics,
negative results, hardware, runtime, and any deviations from baseline parity.
