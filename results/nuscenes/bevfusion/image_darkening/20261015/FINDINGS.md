# Findings: sensitivity to image darkening

## Question and controlled intervention

How does the same pretrained camera–LiDAR detector behave when camera images become artificially darker while geometric inputs remain available?

The pipeline multiplies all six resized camera tensors in float32 by gains 1.00, 0.50 or 0.25 **before data-preprocessor normalization**. Images retain six views, 3 channels and 256 × 704 pixels. Files, camera order, calibration and LiDAR input are unchanged. No training or adaptation occurs. A one-frame smoke check verifies both pre-normalization transformation and unchanged processed LiDAR/calibration across conditions; the full-run pre-hook changes only images.

The experiment uses one official fusion checkpoint (SHA-256 `e68c6c084d8971f4f879751b0ed3981f575c24ee3cf385a1e39197c26a9bb3ab`) and MMDetection3D commit `fe25f7a51d36e3702f961e198894580d83c4387b`, seed 20261002, batch size one, evaluation mode, autocast/TF32 disabled and cuDNN benchmarking disabled. The same 600 ordered tokens cover four selected frames per validation scene. nuScenes-devkit 1.2.0 uses `detection_cvpr_2019` and exactly the same selected ground truth; 12,318 filtered boxes are present per condition.

## Aggregate results

Clean / gain 0.50 / gain 0.25 mAP ×100 are **68.1300 / 67.2352 / 64.5150**; NDS ×100 are **71.2886 / 70.8179 / 69.1982**.

Relative to the fresh clean run, gain 0.50 changes mAP by **−0.8948 points** and NDS by **−0.4707 points**. Gain 0.25 changes them by **−3.6150** and **−2.0904 points**. Aggregate degradation is larger for the stronger perturbation at these tested gains. The experiment does not establish a continuous response curve between the tested levels.

## Class and error behaviour

At gain 0.25 all ten class APs decrease. The largest losses are trucks **−6.1628 points**, barriers **−5.5772**, trailers **−5.5073** and bicycles **−5.4559**. Cars decrease by **−1.1351** and pedestrians by **−1.6279 points**.

Gain 0.50 has a mixed class-level effect: barriers decrease by **−3.1584 points** and bicycles by **−2.2490**, while bus, motorcycle and traffic-cone AP increase slightly (**+0.2626 / +0.2294 / +0.1833 points**). These small increases are descriptive; no significance or mechanism is established.

True-positive errors do not all worsen together. Translation and scale errors increase at both darkening levels. Orientation error decreases slightly at gain 0.50 and increases at gain 0.25; velocity error decreases slightly at both levels. These metrics depend on matched predictions and should not be interpreted as improvements for identical object detections without object-level analysis.

## Interpretation and limits

The measured aggregate loss demonstrates sensitivity to this image perturbation despite unchanged LiDAR input. It does not isolate how much LiDAR compensates for darkness: that would require an additional matched camera-only or other controlled comparison. No automatic sensor fallback is demonstrated.

Multiplying encoded pixel values is not an equivalent reduction in physical illumination. No shot/read noise, exposure adaptation, blur, weather or camera response model is simulated. Results are not real-night performance and gains do not represent lux levels. Four frames per scene are a subset with different weighting from full validation. No scene-bootstrap confidence intervals or significance tests have been computed. Six camera slots and the same checkpoint remain active; there is no sensor-cost saving or runtime claim.

The fresh clean baseline differs very slightly from the preceding camera-availability result, so darkening differences are calculated exclusively against this study's own clean run. No cause for the cross-run difference is inferred.

## Preservation and verification

All three 600-frame prediction files were SHA-256 and token checked on Drive before the compact archive was built. Inference and evaluation logs, source code, configs, metrics and official curves are preserved. Two early metadata-comparison bugs stopped the smoke check before prediction: printed dictionary ordering, then an unsupported Python class value. The corrected canonical comparison passed; both failed attempts remain in the evidence.

On GitHub import the ZIP receipt and all 119 manifest-listed members were verified (120 archive members including the manifest). Class AP means, mAP, NDS and differences were recomputed from stored summaries. This is evidence/arithmetic verification, not a fresh local ground-truth replay. Raw predictions and sensors remain separately on Drive.
