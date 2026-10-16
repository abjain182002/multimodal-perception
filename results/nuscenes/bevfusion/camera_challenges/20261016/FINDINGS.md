# Findings: synthetic blur, noise and calibration challenges

## Question and design

How does the same pretrained camera–LiDAR BEVFusion pipeline respond to controlled image and calibration perturbations? Seven conditions use the same frozen 600 nuScenes validation frames, four from each of 150 scenes, and exactly 12,318 filtered ground-truth objects. The subset overlaps the earlier full-validation evaluation; it is not an independent replication or full-validation robustness benchmark.

The checkpoint SHA-256 is `e68c6c084d8971f4f879751b0ed3981f575c24ee3cf385a1e39197c26a9bb3ab`; source commit is `fe25f7a51d36e3702f961e198894580d83c4387b`. Kaggle inference used one Tesla T4, Python 3.10, Torch 2.1.2+cu121, NumPy 1.26.4, float32 inference without autocast, TF32 disabled and cuDNN benchmarking disabled. No robustness training was performed. CPU evaluation used nuScenes-devkit 1.2.0 and detection_cvpr_2019 with sample iteration restricted to the frozen tokens; official matching and filtering logic remained in place.

Gaussian blur uses sigma 1/2 pixels and kernels 7/13 with reflected boundaries. Additive Gaussian noise uses SD 10/25 in encoded pixel units and clipping to [0,255]. Both operate on six resized/cropped 256×704 images before normalization. One token-specific PCG64 realization is shared across noise severities. Calibration uses common positive 1°/3° yaw about the LiDAR +Z axis: C2L'=Rz C2L, L2C'=inverse(C2L'), L2IMG'=K4 L2C'. This models camera-rig-to-LiDAR frame error, not rotation around each optical centre. Points, intrinsics, augmentation matrices, GT and output frame remain unchanged.

## Aggregate results

| Condition | mAP ×100 | NDS ×100 | Δ mAP points | Δ NDS points |
|---|---:|---:|---:|---:|
| Clean | 68.1018 | 71.2767 | 0 | 0 |
| Blur sigma 1 | 67.3572 | 70.9221 | −0.7446 | −0.3546 |
| Blur sigma 2 | 65.8820 | 69.9316 | −2.2198 | −1.3451 |
| Noise SD 10 | 66.1922 | 70.1681 | −1.9096 | −1.1086 |
| Noise SD 25 | 63.1355 | 68.5923 | −4.9663 | −2.6844 |
| Calibration yaw +1° | 67.4287 | 70.8153 | −0.6731 | −0.4614 |
| Calibration yaw +3° | 64.6339 | 69.3172 | −3.4680 | −1.9595 |

Each stronger tested intervention causes a larger aggregate mAP/NDS loss within its family. SD 25 noise has the largest loss among these particular settings. Severity scales are not matched across families; these results do not rank real-world hazards or establish a continuous severity-response curve. Only this study's fresh clean baseline is used for differences. Earlier studies have slightly different clean values; their cause and significance are unestablished.

## Class behaviour and interpretation

Blur sigma 2 and both noise settings decrease all ten class APs. Bicycle AP loses 7.2524 points under sigma 2 blur; SD 25 noise reduces trailer AP by 11.2171 points and truck AP by 9.9861. Yaw +3° reduces motorcycle AP by 9.7719 and bicycle AP by 6.9873 points, while trailer AP increases approximately 0.7 points. Thus aggregate deterioration is not equivalent to deterioration in every class. Small positive changes are descriptive, not demonstrated improvements or evidence of a mechanism.

TP errors also do not change uniformly. For example, yaw +3° increases translation/orientation errors but slightly lowers scale/attribute errors. These statistics depend on matched predictions; they do not demonstrate improvement on identical detected objects.

The supported conclusion is sensitivity of this pretrained fusion pipeline to the declared image and extrinsic errors even with unchanged LiDAR input. There is no camera-only comparison here, no causal measurement of the amount of LiDAR compensation and no demonstrated automatic fallback.

## Verification and limitations

- CPU synthetic-image tests and 10,800 annotation-camera transform checks passed, followed by one actual common model sample under all seven conditions.
- Full-run input ledgers record image, point and metadata hashes for all 4,200 predictions. Original input hashes and processed points match across conditions; image hashes are unchanged in calibration conditions and metadata is unchanged in blur/noise conditions.
- The first Kaggle temporary environment was lost between sessions. The final setup, dependency repair and strict checkpoint load are recorded. Missing `mmdet` in the combined installation was repaired before inference. A failed smoke attempt cleared the sweep prefix; the corrected attempt checks all 9,600 parsed sensor paths. Both attempts remain preserved.
- Separate Swin pretraining initialization was disabled to avoid a redundant download. The complete official fusion checkpoint loaded strictly before predictions; the initialization warning does not describe the inference weights.
- Raw predictions were exported, rehashed and verified on Drive in `camera_challenges600_inference_export.zip`: 135,429,635 bytes; SHA-256 `7bd97d3032515922d12e770b663e226bacc8e338e096182b9f600f063d32392a`.
- Ground-truth metadata was restored from the verified original backup; all condition token sets and filtered class counts agree. Metric files, source and evaluation logs were verified on Drive.
- On import, the compact ZIP SHA-256 and every manifest-listed member were checked; all 270 original members are preserved byte-for-byte. Stored mAP/NDS arithmetic, baseline differences and input ledgers were checked. This import is not a fresh GT replay or a rehash of raw predictions absent from the compact archive.
- Synthetic Gaussian blur is not a full motion/defocus model; additive noise is not a physical sensor model. Only two positive common-yaw offsets and one fixed noise realization per frame were tested. There are no joint corruptions, multiple perturbation seeds, confidence intervals or significance tests.
- No sensor-price, compute-saving, physical sensor-count optimum, robot safety, real-night/weather performance or physical deployment claim follows from this experiment.

Remaining supervisor work includes critical literature/failure-handling synthesis, sourced monetary scenarios and resolution of comparable camera-only 3D/physical multi-LiDAR scope. The bounded planned corruption experiments are complete; the entire supervisor agenda is not yet complete.
