# Paired BEVFusion nuScenes validation-subset result

Evaluation completed on 2 October 2026 on the same frozen 1,231 samples from
31 complete official validation scenes. This covers 20.45% of the 6,019 official
validation samples. Selection was based on complete sensor availability in
verified archive parts 01–02, not random sampling; part 03 was excluded.

| Pretrained pipeline | mAP × 100 | NDS × 100 |
|---|---:|---:|
| LiDAR | 64.2114 | 69.2087 |
| Camera + LiDAR | 67.8147 | 71.1543 |
| Difference (points) | +3.6033 | +1.9457 |

The camera–LiDAR checkpoint pipeline scored higher on both aggregate metrics
in this subset. This compares two official pretrained pipelines; differences
cannot be attributed solely to adding cameras. It is not a controlled camera
ablation, a full-validation reproduction, or evidence of statistical significance.
Do not extrapolate these scores to the remaining scenes or compare them with
KITTI AP40 as though the metrics and evaluation tasks were interchangeable.

Both checkpoints passed strict loading. Ordered dataset tokens were checked;
both evaluators confirmed exact prediction/ground-truth token equality. Inputs
included six camera keyframes and LiDAR with up to nine preceding sweeps.
Official detection filtering retained 32,276 ground-truth boxes, with every
one of the ten classes represented. The class-support figure records these
post-filter counts rather than the dataset loader's unfiltered class counts.

Figure captions:
- comparison.svg / .png / .pdf: official subset mAP and NDS multiplied by 100,
  on a zero-to-100 axis; both pipelines use the same frozen subset.
- class_support.svg / .png / .pdf: ground-truth box counts after official
  distance, point-support and bike-rack filtering; counts do not measure accuracy.

Reproduce with `python3 -m scripts.plot_bevfusion_subset_results` (matplotlib).
The exact headline values, checkpoint identities and reported output hashes are
in evaluation_summary.json. This compact summary is transcribed from the supplied
Colab execution output. The original metric JSON files, predictions, official
PR/TP curves, scripts and audits were verified on Drive by the execution workflow;
they were not independently opened or imported into this Git commit. Figure hashes
verify the derived figures, not those original evaluation files.

Full evidence remains in the project Drive folder:
`Major_Project/nuscenes_full_validation/bevfusion_subset_validation/`.
Full nuScenes validation remains pending. Mini results are a separate pilot;
this subset result does not replace the mini record or imply full coverage.
