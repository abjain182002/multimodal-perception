# One real CLOCs training frame: KITTI Car

## Frozen inputs and interpretation

- Use the original SECOND `train.txt` and `val.txt` with `configs/splits/kitti_second_split.json`: 3,712/3,769 nonoverlapping frames. Frame `000000` is in **train**; the prior `000001` structural example is in **validation** and cannot be used to develop the fusion model.
- `results/pointpillars_evidence_manifest.json` pins the PointPillars checkpoint and config. The existing `000001` structural NPZ lacks real 3D candidate boxes; its all-negative labels are **not** training targets.
- The complete GT audit is `results/kitti_clocs_gt_audit.json`: 28,374 range-filtered Car boxes over all 7,481 labelled frames.

## Colab: one dense PointPillars export

From this repository's root on a CUDA runtime, set the four paths for the mounted KITTI data, pinned MMDetection3D checkout, pinned checkpoint and original CLOCs split directory. Store the large output somewhere durable **outside Git**; transfer the file to the Mac without changing its bytes.

```bash
python -m scripts.export_clocs_dense_train_frame \
  --dataset "$KITTI_TRAINING" --frame-id 000000 \
  --config "$MMDET_ROOT/configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py" \
  --checkpoint "$POINTPILLARS_CKPT" \
  --train "$SECOND_SPLITS/train.txt" --val "$SECOND_SPLITS/val.txt" \
  --output "$EXTERNAL_OUTPUT/000000_dense_candidates.npz"
```

The export rejects mismatched splits/checkpoints/config names and unavailable CUDA. Its NPZ retains the original pre-NMS candidate order, camera and LiDAR boxes, 2D projections, scores, directions, distances and input provenance hashes. Record its printed SHA-256 when transferring it.

## Mac: real YOLOv8x associations and 3D IoU targets

Set `YOLO_WEIGHTS` to the locally frozen `yolov8x.pt`. Copy the Colab NPZ to the `--candidates` location below and run this from the repository root in `azitt310`:

```bash
python -m scripts.build_clocs_yolo_train_frame \
  --dataset "$HOME/Major_Project/datasets/KITTI/training" --frame-id 000000 \
  --candidates "$HOME/Major_Project/model_outputs/clocs_stage5_checkpoints/000000_dense_candidates.npz" \
  --yolo-weights "$YOLO_WEIGHTS" --device mps \
  --train "$HOME/Major_Project/external/CLOCs/second/data/ImageSets/train.txt" \
  --val "$HOME/Major_Project/external/CLOCs/second/data/ImageSets/val.txt" \
  --rows "$HOME/Major_Project/model_outputs/clocs_stage5_checkpoints/000000_clocs_rows.npz" \
  --output "$HOME/Major_Project/model_outputs/clocs_stage5_checkpoints/000000_real_targets.npz" \
  --audit results/clocs_real_frame_000000_audit.json
```

The Mac step checks frame and archive provenance, recomputes all 2D/3D feature rows, computes camera-coordinate rotated **3D IoU**, applies positive `>=0.70`, negative `<=0.50` and ignored thresholds, and reloads the saved example with its content hash verified. Inspect real Car GT and positive/negative/ignored counts. Frames with no retained Car GT can legitimately be all-negative; one frame is only an integrity gate, **not** a trained model or KITTI AP result.

Commit the inspected small audit JSON to GitHub as a separate experimental milestone. Keep KITTI data, NPZ files and model weights in durable external project storage; record their SHA-256 hashes in the audit. Before any actual fusion training, compare sampled IoUs against the pinned SECOND/CLOCs evaluator and confirm a real positive signal on multiple train frames. Validation stays frozen until the final evaluation.

## Progress and next geometry gate (26 September)

Train frame `000000` is a valid all-negative control; `results/clocs_real_frame_000000_audit.json` records zero Car GT and zero YOLO Car detections. The independent train frame `000149` has 10 retained Car GT, 9 YOLO Car detections, 179 positive/106,798 negative/159 ignored candidates, and a maximum computed 3D IoU of 0.929133. See its candidate export manifest and `results/clocs_real_frame_000149_audit.json`. These are real examples, not a trained CLOCs model.

Before expanding the training corpus, run the independent **pinned MMDetection3D KITTI evaluator** on the `000149` candidate export in the already prepared CUDA environment. The standalone script checks input hashes, re-derives full-frame target counts from original KITTI labels, then compares representative positive, ignored, overlapping-negative and zero-IoU candidates with the reference GPU `d3_box_overlap` implementation. It imports this pinned evaluator directly because importing the whole evaluation package also requires unrelated optional dataset SDKs. The pinned KITTI evaluator uses the same camera-coordinate 3D overlap formulation as the SECOND/CLOCs evaluator; record it explicitly as the MMDetection3D comparison, not as a CLOCs run.

```bash
cd "$PROJECT_ROOT"
"/content/pointpillars310/bin/python" -m scripts.audit_clocs_iou_reference \
  --dataset "$KITTI_TRAINING" \
  --candidates "/content/drive/MyDrive/Major Project/cloud transfer/clocs_stage5_checkpoints/000149_dense_candidates.npz" \
  --mmdet3d-root /content/mmdetection3d \
  --audit results/clocs_real_frame_000149_audit.json \
  --output results/clocs_000149_iou_reference_audit.json
```

Set `PROJECT_ROOT` to the checkout containing this script and `KITTI_TRAINING` to the restored directory containing `000149.txt` in both `label_2/` and `calib/`. A `PASS` report with numerical differences, reference source SHA and candidate provenance is the exit criterion. The report and a measured IoU parity PNG are written even if the comparison fails; investigate failures before training. Review and commit the small report and figure once run and checked. A single positive frame still does not meet the later multi-frame training-corpus gate.

### Native CUDA evaluator blocker and independent CPU check

On the available Colab T4 runtime, loading the pinned KITTI evaluator directly bypassed an unrelated missing Lyft SDK, but the resulting native process exited with signal 11 before writing a report. This is an **unresolved reference-runtime crash**, not an observed IoU disagreement. Do not describe that evaluator as passed. `scripts/audit_clocs_iou_scipy.py` supplies an independent CPU geometric check using SciPy/Qhull halfspace intersections on the same hash-verified `000149` inputs; it saves a separate report and correctly labelled measured parity PNG. Its `PASS` cannot be substituted for a completed pinned MMDetection3D/SECOND evaluator run. Retain the GPU runtime blocker and resolve it before claiming official evaluator agreement.
