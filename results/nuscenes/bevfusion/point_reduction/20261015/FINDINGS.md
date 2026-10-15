# Findings: matched LiDAR point reduction

## Question

How do two established pretrained pipelines respond to the same reduction in available LiDAR points, and does the measured fusion advantage persist?

The comparison uses 600 previously frozen validation frames (four per each of 150 scenes), six fresh inference conditions and exactly the same 12,318 filtered ground-truth boxes. Source commit `fe25f7a51d36e3702f961e198894580d83c4387b`, Torch 2.1.2+cu121, NumPy 1.26.4, seed 20261002, batch size one, evaluation mode, autocast/TF32 disabled and cuDNN benchmarking disabled are recorded. The two model checkpoint hashes are frozen in the protocol; this project did not train these detectors.

## Controlled intervention

The source point tensor contains the keyframe plus loaded preceding LiDAR sweeps. For each token a SHA-256-derived seed initializes NumPy PCG64. The first floor(fraction × N) entries of one permutation select a nested subset (at least one point); sorting selected row indices restores original order. No retained coordinates or attributes are changed. Fractions are 1.00, 0.50 and 0.25. The modification happens before data preprocessing, voxelization and fusion depth construction; images, calibration and original sensor files are unchanged.

The six-condition smoke check tests one shared frame and finite predictions. Full inference records original point hashes, selected point hashes, index hashes, counts, image hashes and metadata hashes for each of 600 frames per condition. These match between pipelines as appropriate, with image/metadata consistency across fractions within each pipeline. An index hash cannot reconstruct raw sensor values: raw sensor verification is a separate provenance record.

## Aggregate findings

| Retention | LiDAR mAP ×100 | Fusion mAP ×100 | Fusion minus LiDAR (points) |
|---|---:|---:|---:|
| 100% | 64.3076 | 68.1302 | +3.8226 |
| 50% | 60.5637 | 65.7931 | +5.2294 |
| 25% | 51.0381 | 60.2341 | +9.1960 |

At 50% retention, mAP changes from each model's own baseline by **−3.7439 points (LiDAR-only)** and **−2.3371 (fusion)**. At 25% retention the changes are **−13.2695** and **−7.8961 points**, respectively. NDS losses at 25% are **−8.3353** and **−5.8101 points**. Both pipelines deteriorate; fusion's aggregate loss is smaller at the two tested reduced settings. Its measured advantage grows as the retained fraction decreases in this experiment.

These comparisons describe two separate trained pipelines, not a causal isolation of the camera branch. In particular, the fusion depth transform also uses LiDAR points, so point loss changes both its LiDAR feature path and depth information supplied to its image path. No automatic sensor fallback or invariant performance is demonstrated.

## Class behaviour

At 25% retention every class AP decreases for both models. LiDAR-only motorcycle and bicycle losses are **−29.3444 / −26.4360 points**; fusion losses for those classes are **−13.7552 / −17.1919**. Both still have substantial vulnerability. At 50%, a smaller overall fusion loss does not mean a smaller loss in every class: bus AP falls about **3.04 points in fusion versus 0.33 in LiDAR-only**. Fusion construction-vehicle AP increases about **0.57 points** at that setting. Low class support and the absence of uncertainty estimates limit interpretation of small changes.

## Limits and engineering implications

- Uniform sampling is synthetic point thinning, not a calibrated model of beam count, range-dependent dropouts, occlusion, weather, sensor failure or a lower-cost physical scanner.
- Temporal sweeps come from one physical sensor; no multi-LiDAR experiment is performed.
- One fixed random selection realization is used per token. No multi-seed perturbation study or scene-bootstrap interval has been computed.
- The 600 frames overlap the original official validation evaluation and earlier robustness studies; they are not independent test replications or full-validation results.
- Use this study's own fresh baselines for loss calculations. Very small cross-run baseline differences exist; no cause or significance is inferred.
- No latency, power, hardware bill of materials or deployable robot performance is established by these accuracy runs.

The supported engineering conclusion is that the evaluated fusion pipeline retains higher aggregate detection accuracy under matched synthetic point loss, while remaining sensitive to LiDAR degradation. A robust deployment argument also requires sensor geometry, calibration, compute, cost and application-specific validation.

## Import verification

The uploaded archive's SHA-256 and all 217 manifest-listed files were checked (218 members including the manifest). All original member bytes are preserved. Input ledgers and protocol hashes were checked; deterministic index-selection hashes were recomputed for all 600 tokens. Class AP, mAP, NDS, own-baseline changes and same-retention pipeline differences were recomputed from the stored summaries within 1e-12. These are stored-evidence checks, not a fresh ground-truth replay or raw-prediction rehash. Original Colab verification and Drive preservation are recorded in the logs and resume checkpoint.
