# Findings: camera availability on the frozen 600-frame subset

## What was tested

The experiment uses 600 annotated keyframes: four timestamp-ordered positions at approximately 20%, 40%, 60% and 80% of each of the 150 official validation scenes. The rule and tokens were frozen before the new predictions. Every condition uses the same tokens and 12,318 filtered ground-truth objects. Nine preceding LiDAR sweeps are available per selected frame; these are temporal inputs from one physical LiDAR, not multiple LiDAR sensors.

A separate pretrained LiDAR-only checkpoint is the reference pipeline. The three fusion conditions share the exact same pretrained fusion checkpoint. Missing views are represented by zero normalized image tensors (mean-colour inputs), with camera geometry and all six computational view slots retained. The mask is applied after preprocessing. The smoke check confirms masked channels and unchanged retained views, LiDAR inputs and calibration geometry on one common sample. The inference hook checks camera channel identity and image masking for every newly inferred frame.

## Results

| Condition | mAP (%) | NDS (%) | Δ mAP vs intact fusion (points) | Δ NDS vs intact fusion (points) |
|---|---:|---:|---:|---:|
| LiDAR-only checkpoint | 64.31 | 69.44 | -3.82 | -1.85 |
| Fusion: six camera views | 68.13 | 71.29 | +0.00 | +0.00 |
| Fusion: front/back retained; four masked | 63.63 | 68.76 | -4.50 | -2.53 |
| Fusion: front masked; five retained | 64.84 | 69.44 | -3.29 | -1.85 |


Intact fusion improves mAP by 3.8242 points and NDS by 1.8484 points over the separate LiDAR-only checkpoint. This baseline gain has the same direction as the earlier full-validation comparison, but the 600 frames overlap that validation set; it is not independent replication evidence.

Retaining front/back views and masking four others reduces mAP by 4.5049 points and NDS by 2.5275 points against intact fusion. It also scores below the LiDAR-only pipeline: −0.6807 mAP points and −0.6791 NDS points. This indicates that the existing fusion model does not automatically fall back to the separate LiDAR-only model's performance when deprived of image information.

Masking only CAM_FRONT reduces mAP by 3.2947 points and NDS by 1.8483 points against intact fusion. Its mAP remains 0.5295 points above LiDAR-only. Its NDS exceeds LiDAR-only by only 0.0000471 points, an effectively equal numerical result, not a meaningful demonstrated advantage.

Both masked conditions reduce AP for every one of the ten classes relative to intact fusion. The largest drop is for traffic cones: −11.6194 points with four views masked and −6.2151 with front masked. Motorcycle drops are −5.9688 and −5.8782 points respectively. Bus AP drops more under front-only masking (−4.1961) than under the front/back condition (−2.0880), showing that camera arrangement matters and class outcomes are not strictly ordered by available camera count. These observations do not identify the spatial or visual cause; sector-level matching would be needed.

## Evidence checks

Both checkpoint hashes match the frozen full-validation protocol, and strict model loading passed. All 9,600 selected sensors were restored with archive and sensor hashes checked. Annotation order matches all 600 frozen tokens. The nuScenes-devkit 1.2.0 evaluator was used with detection_cvpr_2019 configuration. A wrapper restricts the ground-truth loader's sample iteration to the selected frames while retaining the SDK tables; matching, distance/point/bike-rack filtering and metric computation use the existing evaluator. All four prediction/GT token sets and filtered class counts match.

The compact ZIP SHA-256 and 158 manifest member hashes were checked on import. Its two additional members are the manifest and raw-prediction receipt (160 total). Stored mAP was recomputed as the mean of per-class mean AP; stored NDS was recomputed from mAP and the five TP errors, with agreement within 1e-12. These are integrity and arithmetic checks, not a second evaluator or fresh ground-truth replay. All three new figures were visually inspected.

## Practical limits and remaining work

- This is a selected 600-frame subset, not full-validation accuracy. Equal frames per scene differs from frame-weighted full validation. The selection covers all scenes but does not establish balanced weather, illumination, geography or object-class representation.
- The experiment measures inference-time missing-view robustness. It does not measure a model trained specifically for two cameras, physically disconnecting sensors, or an optimal two-camera arrangement.
- Masking leaves all six views in the computation. No hardware cost saving, latency saving, or memory saving follows from these accuracy results.
- The LiDAR-only versus intact-fusion comparison uses different trained pipelines; camera masking within fusion is the controlled intervention.
- No confidence intervals or significance test were computed. Frames within scenes are correlated, and low-support class estimates require caution.
- The runtime used nvcc 12.1.66; the prior inference setup reported 12.1.105. The change is recorded; core GPU, custom pooling/voxelization and model smoke checks passed. Other transitive packages may also differ, and full environment records are preserved.
- Still required for cost/performance claims: declare and measure latency and peak memory on a common setup, source hardware costs, and assess feasibility of a separate camera-only 3D model. A multi-LiDAR experiment requires suitable multi-sensor data and models.

The supported conclusion is that this pretrained fusion pipeline benefits from intact camera inputs and loses accuracy under the specified missing-view conditions. It cannot by itself establish which physical sensor arrangement offers the best deployment value.
