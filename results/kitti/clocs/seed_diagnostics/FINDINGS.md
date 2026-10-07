# CLOCs seed and epoch diagnostics — 1 October 2026

All four runs evaluated 3,769 validation frames with Car strict IoU 0.70 AP40.
PointPillars reproduced all nine frozen baseline metrics exactly.

At epoch 12, Moderate 3D AP40 was 79.8616, 76.2666 and 78.7616 across the
three seeds, against PointPillars at 79.1299. Only one seed improved Moderate;
all three reduced Hard AP. These are descriptive results, not significance tests.

For seed 20260928, extending training from epoch 12 to 24 increased development
candidate AP from 0.877935 to 0.883568. Adam state was restored, the learning rate
remained 0.002, and no scheduler was used. Epoch 24 was development-selected.
Moderate 3D AP40 nevertheless fell from 79.8616 to 79.4874; Hard fell from
74.8719 to 74.7797. Candidate-AP improvement did not translate into improved
3D detection AP40 in this extension. A training plateau remains unestablished.

The study used 96 optimization and 32 development frames. The original seed
selection used development AP before its full-validation run. Validation has
subsequently been inspected iteratively and is not an untouched final test.
The extension checkpoint lacked saved RNG states, so continuation is not
claimed to reproduce an uninterrupted run bitwise.

These results support a limitation of this small-data implementation and
protocol; they do not establish that CLOCs or camera–LiDAR fusion generally fails.
The epoch-24 run is a one-seed diagnostic, separate from the three-seed study.

Figure captions:
- three_seed_and_epoch24_3d_deltas.svg: Fusion minus paired PointPillars 3D AP40
  in AP points. Epoch-24 continuation is shown separately from epoch-12 seeds.
- epoch12_to24_candidate_ap.svg: Development candidate AP during continuation.
  The truncated vertical axis is explicit; this measure is not detection AP40.

