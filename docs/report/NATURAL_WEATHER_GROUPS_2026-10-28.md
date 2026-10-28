# Natural rain/night strata: results, coverage and interpretation

## Question and design

This post-hoc extension asks how two existing pretrained LiDAR and camera-LiDAR pipelines compare within scene-description strata of the complete previously evaluated nuScenes validation population. It uses all 6,019 frames in 150 official validation scenes and the same saved predictions, ten-class configuration and SDK filtering. There is no new inference or training. Prior aggregate and selected-case results were already known; the new group definitions and membership were fixed before subgroup metrics were accessed.

Descriptions define four disjoint groups: neither condition mentioned (111 scenes / 4,449 frames / 15 logs), rain only mentioned (24 / 968 / 2), night only mentioned (12 / 482 / 1), and night plus rain mentioned (3 / 120 / 1). All 150 descriptions were reviewed, with no literal-group amendment. These are metadata strata: absence of a mention does not establish dry weather or daylight, and a scene description does not confirm the same weather in every frame. There are 18 unique logs overall; all 15 night scenes share one log across the two night groups.

The CPU evaluation first replays the full split for each pipeline and requires original summary and detailed metrics to agree within 1e-12, excluding evaluation time. Both pass: 32,451 numeric values per pipeline, maximum absolute discrepancy 8.881784197001252e-16. The smaller evaluation-only metadata preserves all annotations and temporal links, retaining validation keyframes and their poses. Agreement with the original full-split results supports its use for this evaluator; it does not make it an inference dataset or independently verify evaluator correctness.

## Ten-class subgroup results

All scores below are multiplied by 100. These are SDK metrics on subsets, not additional official full-validation benchmark scores. Differences are fusion minus LiDAR, in points.

| Description group | LiDAR mAP | Fusion mAP | mAP difference | LiDAR NDS | Fusion NDS | GT classes present |
|---|---:|---:|---:|---:|---:|---:|
| Neither mentioned | 64.10 | 67.68 | +3.57 | 69.06 | 70.84 | 10 |
| Rain only mentioned | 64.39 | 70.65 | +6.26 | 70.92 | 73.86 | 10 |
| Night only mentioned | 45.09 | 46.32 | +1.23 | 46.57 | 47.60 | 7 |
| Night and rain mentioned | 26.36 | 27.28 | +0.93 | 24.41 | 26.90 | 4 |

Fusion has higher aggregate scores within all four observed groups. This does not mean every class or frame improves. Rain-only barrier AP decreases by 0.32 points; night-only motorcycle and barrier AP decrease by 2.36 and 1.01 points respectively. The pipelines were separately pretrained, so these comparisons do not isolate the causal contribution of cameras.

## Class coverage changes the interpretation

Night-only contains no filtered GT bus, trailer or construction-vehicle instances. Night/rain additionally contains no filtered bicycle, traffic-cone or barrier instances. Their absent classes receive zero AP under the retained ten-class SDK calculation. Consequently, low ten-class subgroup mAP is partly a property of coverage and cannot be presented as a measured weather-induced loss. With only four GT classes, even perfect AP on every represented class would yield ten-class mAP x100 of 40.

A predeclared sensitivity summary uses the same four GT-represented classes across every group: car, truck, pedestrian and motorcycle. It averages their class APs; it is not official ten-class mAP and does not adjust NDS or remove within-class distribution differences.

| Description group | LiDAR four-class macro AP x100 | Fusion four-class macro AP x100 | Difference |
|---|---:|---:|---:|
| Neither mentioned | 74.98 | 78.67 | +3.69 |
| Rain only mentioned | 73.08 | 80.30 | +7.22 |
| Night only mentioned | 77.56 | 78.26 | +0.69 |
| Night and rain mentioned | 65.89 | 68.21 | +2.31 |

Night-only LiDAR macro AP on this fixed class set is higher than the neither-mentioned group, despite much lower ten-class mAP. This illustrates why the raw ten-class bars alone do not establish general night degradation. The night/rain four-class mean remains lower than the other strata, but location, object composition, distance, visibility and support are uncontrolled. Neither a causal rain/night effect nor broad environmental generalisation is established.

## Pedestrians and shared difficulty

| Description group | Filtered pedestrian GT instances across frames | LiDAR pedestrian AP x100 | Fusion pedestrian AP x100 |
|---|---:|---:|---:|
| Neither mentioned | 22,411 | 87.14 | 88.40 |
| Rain only mentioned | 1,003 | 79.31 | 81.69 |
| Night only mentioned | 64 | 60.87 | 61.73 |
| Night and rain mentioned | 22 | 17.28 | 20.59 |

The night/rain pedestrian result identifies low observed performance that is not explained by the absence of other classes from the macro average. Fusion improves pedestrian AP by 3.31 points there, while both pipelines remain low. However, 22 annotation instances across frames are not 22 independent people or environments. The same night/rain stratum has only four truck and 17 motorcycle GT instances, so equal class weighting can magnify results based on sparse support. The selected scene-1067 shared pedestrian miss provides a complementary local diagnostic, not an independent representative validation sample or proof of a weather mechanism.

## Limits and implications

These are descriptive results, with no confidence intervals, p-values or statistical-significance claims. Frames within scenes are correlated, and scenes can share acquisition logs. One night log and two rain-only logs sharply limit environmental generalisation; treating frames as independent or using scene resampling alone would not solve that limitation. No rain-intensity measurements, image-level weather labels, matched weather counterfactuals, or control of within-class difficulty are available in this analysis.

The contribution is a reproducible audit showing how aggregate pipeline gains coexist with class-specific regressions, class-coverage effects and low pedestrian AP in a small night/rain stratum. A controlled synthetic fog study can complement this observational analysis by holding frames and checkpoints fixed, but it remains synthetic evidence. Snow, glare, wind, storms and flooding require literature-supported limitations and separately designed validation; this study establishes no performance under them.

## Evidence and verification

The [completed evidence](../../results/nuscenes/bevfusion/weather_groups/20261028/completed/README.md) includes exact group/class CSVs, replay audits, a coverage-aware figure and the unchanged 2,156,223-byte source ZIP containing all 69 manifest-listed payloads plus its manifest. ZIP SHA-256: `75126afda2354f95f9402fccfa37100fb45d3f956cb1d5600cfb6fdf2c289227`. The supplied Colab receipt reports a verified Drive copy and the uploaded ZIP matches that receipt; Drive was not accessed directly during import.

Local verification checks all payload sizes/hashes, reconciles 64,902 full-replay/reference numeric values, recomputes 400 AP values from saved precision-recall curves, checks ten-class mAP/NDS arithmetic, and verifies all eight group rows, 80 class rows, paired differences and the partition of filtered GT counts. This is evidence import and arithmetic reconciliation, not another local GT evaluation. Original numerical outputs are unchanged.
