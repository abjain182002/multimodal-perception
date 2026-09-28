# CLOCs 128-frame internal development protocol

This is an adapted CLOCs experiment, not a reproduction of original paper scores.
PointPillars and YOLO26x remain frozen. Only the fusion network is optimized.

## Data and selection

Use the verified 128 KITTI training frames. Sort eligible frame IDs by SHA-256
of `20260928:frame_id`; first 32 form internal development, remaining 96 optimization.
The 16 frames used in earlier pilot analysis are always in optimization. No KITTI
validation frames are used. This is frame-disjoint, not a verified scene-disjoint
split. Internal development is for tuning and is not an independent test set.

## Controlled comparison

First run ratio 0 (all valid candidates), then ratio 8. Both runs use the same
initialization seed, epoch orders, architecture, learning rate 0.002 and 12 epochs.
The best epoch is selected by full development candidate AP. Sampling is a
hypothesis, not a guaranteed improvement. Ratios 4 and 16 are optional subsequent
development experiments, to be reported if tried.

Sampling retains all positives and no ignored candidates. Negative budget is
ratio times positive count, capped by available negatives. Zero-positive frames
use 256 negatives at most. Half (rounded down) are highest PointPillars-score
negatives; the remainder are sampled without replacement from remaining negatives.
The seed depends on experiment seed, epoch and frame. High PointPillars score is a
hardness proxy, not online fusion-loss mining. Sampling applies to candidate loss
after aggregation; all association rows are forwarded. Thus this implementation
does not promise GPU memory savings. Loss is mean focal loss over selected
candidates per frame; sampling changes the training objective. Selected training
loss magnitudes should not be compared directly across methods.

Evaluation uses all nonignored candidates, with candidate average precision and
full-candidate focal loss. These metrics are NOT KITTI 3D/BEV AP40. Detection
post-processing and evaluation on all 3,769 validation frames remain future work.
Calibration, if used, must be selected on internal development before the final
validation run. No rerolling splits based on development performance.

Each run stores split, input hashes, protocol/environment/source hashes, per-epoch
history, latest checkpoint, best-development checkpoint, CSV and reload check.
Runs require empty output directories. Latest checkpoints are recovery evidence;
automatic mid-run resume is not implemented. Fixed seeds do not guarantee
bitwise equality across different GPU/software stacks. Repeat seeds before
making strong claims. Existing 16-frame scripts/results remain unchanged.

## Attribution

Su Pang, Daniel Morris, Hayder Radha. CLOCs: Camera-LiDAR Object Candidates Fusion
for 3D Object Detection. IROS 2020. https://arxiv.org/abs/2009.00784
Original implementation: https://github.com/pangsu0613/CLOCs
The negative sampler and this development protocol are project adaptations.

## Run from repository root on Kaggle

Use the notebook's GPU-enabled Python used successfully for the earlier CLOCs
pilot. MMDetection3D is not needed for training on these cached features.

```bash
python -m scripts.train_clocs_128 \
  --data-dir /kaggle/working/clocs_yolo26_train_128 \
  --batch-manifest results/clocs_train_export_128_manifest.json \
  --train-ids /kaggle/input/datasets/abhishekjain18/clocs-train-export-128-tar-gz/splits/train.txt \
  --val-ids /kaggle/input/datasets/abhishekjain18/clocs-train-export-128-tar-gz/splits/val.txt \
  --output-dir /kaggle/working/clocs_128_ratio0_seed20260928 \
  --negative-ratio 0 --epochs 12 --device cuda
```

Then repeat with `--negative-ratio 8` and a new output directory ending `ratio8_seed20260928`.
