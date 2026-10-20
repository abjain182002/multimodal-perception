# nuScenes checkpoint-reference audit — 20 October 2026

**Bounded audit complete; the cause of the remaining score discrepancy is unresolved.** Both upstream checkpoint downloads match the full hashes recorded for this project's inference. All six saved model configurations and test pipelines (two models × three storage batches) match the pinned MMDetection3D reference. The fresh full-validation replay independently establishes same-evaluator reproduction of the project's preserved metrics; it does not explain the upstream difference.

## Comparison targets and precision

The comparison uses MMDetection3D commit `fe25f7a51d36e3702f961e198894580d83c4387b`, its BEVFusion model table and its linked training/validation logs. It does not substitute another group's BEVFusion architecture or hidden-test leaderboard scores. All score differences below are absolute points after multiplying mAP/NDS by 100.

| Pipeline / metric | Rounded model table | Selected linked-log row | Project full validation | Project minus logged score |
|---|---:|---:|---:|---:|
| LiDAR mAP ×100 | 64.9 | 64.85, epoch 20 | 64.4301 | −0.4199 |
| LiDAR NDS ×100 | 69.6 | 69.56, epoch 20 | 69.2353 | −0.3247 |
| Fusion mAP ×100 | 68.6 | 68.58, epoch 5 | 68.3099 | −0.2701 |
| Fusion NDS ×100 | 71.4 | 71.39, epoch 5 | 71.1352 | −0.2548 |

The log rows are consistent with the rounded table entries. Their printed scores also have limited precision: the four-decimal differences above are arithmetic against rounded log values, not exact upstream error estimates. The original [table-based comparison](../../../benchmarks/BENCHMARK_COMPARISONS.md) remains valid at its stated precision. The table does not prove that the published weight file was exported from the identified epoch; checkpoint metadata-to-log-row identity was not inspected. No upstream raw prediction file was evaluated.

Both linked logs explicitly report a validation dataset length of 6,019. The fusion log's `[753/753]` progress is from its distributed execution and should not be described as a 753-sample validation split. All ten logged validation rows are preserved numerically in [upstream_validation_rows.csv](upstream_validation_rows.csv); these rows are not independent experimental replicates.

## Checks and limits

| Item | Verified finding | Limit |
|---|---|---|
| Checkpoint identity | Streamed both published weight URLs on 20 October and computed full SHA-256: LiDAR `2628f933…` (46,332,893 bytes), fusion `e68c6c08…` (167,113,973 bytes). They match all six saved inference protocols. | This checks the currently downloaded bytes against recorded execution hashes, not a new inference run or checkpoint metadata inspection. Full values and URLs: [checkpoint_hash_audit.json](checkpoint_hash_audit.json). |
| Model configuration | Entire resolved `model` dictionaries match the inherited pinned reference for both models in all three batches. | Static settings equality does not prove identical runtime numerics or upstream training-code revision. |
| Inference pipeline | Entire saved `test_pipeline` lists match the pinned reference across all six records. Nine preceding LiDAR sweeps, five point features, empty-sweep padding, close-point removal and range filtering agree. Fusion image processing is 256×704 with fixed resize 0.48, no test-time image rotation or flip. | Equality of settings does not compare every processed tensor with upstream outputs or prove identical annotation conversion/sweep contents. |
| Dataset loader | Only three field differences appear: local data root, per-batch annotation path, and explicit `metainfo.version='v1.0-trainval'`. | Project official token coverage is verified separately. The upstream annotation pickle was not obtained for a byte/content comparison. See [config_differences.json](config_differences.json). |
| Checkpoint loading | Each executed batch script includes strict full-checkpoint loading; saved configuration hashes match their protocols exactly. | This is source/protocol evidence, not a new checkpoint load performed by this audit. |
| Evaluator configuration | Project uses nuScenes-devkit 1.2.0 and `detection_cvpr_2019`, with preserved source hashes. Pinned upstream `NuScenesMetric` defaults to `detection_cvpr_2019`. | The linked logs do not specify their nuScenes-devkit version/source hashes. Exact upstream evaluator-version equality remains unverified. |
| Ground-truth replay | On 20 October both models reproduced all 32,451 compared numeric summary/detail values within 1e-12, excluding evaluation time. | Same-evaluator reproduction; independent implementation correctness and the upstream discrepancy are separate questions. [Replay evidence](../full_validation/replay_20261020/README.md). |

The saved `test_evaluator` configuration remains identical to the pinned config. The executed inference scripts explicitly override its data root and annotation paths for formatting, and full accuracy is evaluated separately by the preserved SDK script. The audit therefore does not mistake the unused original relative paths in the config snapshot for an actual evaluation-path error.

## Differences from the historical logged execution

Matching the pinned configuration is not the same as matching every aspect of the 2023 logged run. The logs record Python 3.8.13, PyTorch 1.11.0+cu113 and eight NVIDIA A100-SXM4-80GB GPUs. Project inference records Python 3.10-era setup, PyTorch 2.1.2+cu121 and one Tesla T4. These are observed differences, not demonstrated explanations of the score gap.

The LiDAR log uses SyncBN in three model normalization settings, whereas the pinned/current project configuration uses BN/BN1d. The fusion log names its depth view-transform field `vtransform`; the pinned/project field is `view_transform`, with matching transform parameters. The logged fusion backbone initialization path also differs; project scripts subsequently load the complete fusion checkpoint strictly. Packed metadata-key lists differ between historical logs and pinned configurations. [logged_config_differences.json](logged_config_differences.json) records these comparisons. None of these observations has been isolated experimentally, so no causal attribution is made.

## Defensible report statement

The pretrained pipelines reproduced the pinned model configuration and inference preprocessing, and both published checkpoint hashes matched the recorded project weights. A fresh same-evaluator CPU replay reproduced the project's full-validation results within 1e-12. However, the project remains approximately 0.42/0.32 mAP/NDS points below the linked LiDAR validation log and 0.27/0.25 points below the linked fusion log, subject to log precision. Historical runtime/configuration differences were identified, while upstream evaluator-version and annotation equivalence remain unverified. The discrepancy is therefore reported as an unexplained reproduction difference rather than attributed to hardware, random seed or preprocessing without a controlled comparison.

The project's within-evaluation fusion advantage remains +3.8798 mAP points / +1.8998 NDS points. This compares two pretrained pipelines; neither the replay nor the reference audit makes it a controlled camera ablation. Future resolution would require the upstream evaluator environment, exact annotation conversion and checkpoint/log provenance, followed by a narrowly controlled reproduction. No further GPU experiment was performed for this audit.

## Primary sources and reproduction

- [Pinned model table and checkpoint/log links](https://github.com/open-mmlab/mmdetection3d/blob/fe25f7a51d36e3702f961e198894580d83c4387b/projects/BEVFusion/README.md).
- [Pinned LiDAR configuration](https://github.com/open-mmlab/mmdetection3d/blob/fe25f7a51d36e3702f961e198894580d83c4387b/projects/BEVFusion/configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py) and [fusion configuration](https://github.com/open-mmlab/mmdetection3d/blob/fe25f7a51d36e3702f961e198894580d83c4387b/projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py).
- [Linked LiDAR execution log](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d_20230322_053447.log) and [fusion execution log](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d_20230524_001539.log).
- [Pinned NuScenesMetric evaluator default](https://github.com/open-mmlab/mmdetection3d/blob/fe25f7a51d36e3702f961e198894580d83c4387b/mmdet3d/evaluation/metrics/nuscenes_metric.py).

Retrieved 20 October 2026. [source_manifest.json](source_manifest.json) records complete retrieved-source hashes plus inspected original evidence-file hashes. Raw upstream logs/configs are obtained from the sources above when regenerating the audit; derived facts and differences are preserved here. Original project evidence is authenticated against the verified 10 October archive SHA-256.

From the repository root, regenerate static comparisons and log rows with the standard-library script:

```bash
python scripts/audit_nuscenes_reference.py --original-archive /path/to/nuscenes_full_validation_evidence_20261010.zip --source-dir /tmp/bevfusion_reference_sources --checkpoint-audit results/nuscenes/bevfusion/reference_audit_20261020/checkpoint_hash_audit.json --output /tmp/bevfusion_reference_audit
```

This command uses the preserved checkpoint download audit; it does not redownload the weights. To independently refresh it, stream each weight URL from that JSON through SHA-256 and compare its size/hash before supplying the resulting record. The original audit performed that streaming check; no bulk weights are added to Git. The configuration reader accepts only the simple literal/name/arithmetic/dictionary forms needed by these pinned files, merges inherited dictionaries for the inspected branches, and skips the fusion config's deletion of a training-only custom hook. It is not a general MMEngine configuration resolver.
