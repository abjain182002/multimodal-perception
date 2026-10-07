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
The exact headline values and reported output hashes are in evaluation_summary.json.
On 7 October 2026, the original evidence archive was opened and all 84 members
were verified against their recorded sizes and SHA-256 hashes. Both prediction
files contain exactly the frozen 1,231 tokens; prediction and metric hashes match
the original evaluation audits. Stored mAP/NDS arithmetic was checked against
the stored class APs and TP scores. This is saved-evidence verification, not a
fresh accuracy evaluation from predictions and ground truth.

Original metric summaries, evaluation/inference audits, resolved configurations,
scripts, frozen tokens, split and scene records, and official summary curves are
now imported under original_evidence/. The complete archive manifest and the
new verification receipt record provenance and exclusions. Large predictions,
converted annotation pickle, sensor manifest, logs and other PR/TP curves remain
in the hash-verified source archive. A CPU ground-truth evaluation replay is still
pending. Historical RESUME_NOTE.json predates completed accuracy evaluation.

Full evidence remains in the project Drive folder:
`Major_Project/nuscenes_full_validation/bevfusion_subset_validation/` and
`bevfusion_subset_original_evidence.zip` in its parent folder.
Full nuScenes validation remains pending. Mini results are a separate pilot;
this subset result does not replace the mini record or imply full coverage.

