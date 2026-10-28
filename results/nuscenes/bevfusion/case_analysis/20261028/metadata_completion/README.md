# Ten-class filtering and visibility completion — 28 October 2026

The uploaded recovery file's SHA-256 is `40ebf900154cf8656b0b87c55e9ca3b875908bf972453b0f48e3ddbea7d040c1`. Its 339 original annotations cover cases A–E and two night-labelled candidate frames. The source archive was hash-checked during Colab recovery. This analysis performs no inference.

## Full-frame diagnostic counts at score 0.30

Cells report **TP / FP / FN** across all ten detection classes using the original fixed diagnostic XY matching distance <2 m. These are fixed-threshold case counts, not nuScenes AP/NDS.

| Case | LiDAR clean | Fusion clean | Fusion gain 0.50 | Fusion gain 0.25 |
|---|---|---|---|---|
| A | 22 / 2 / 24 | 33 / 3 / 13 | 30 / 4 / 16 | 26 / 6 / 20 |
| B | 21 / 2 / 17 | 14 / 3 / 24 | 15 / 3 / 23 | 12 / 2 / 26 |
| C | 71 / 1 / 8 | 72 / 1 / 7 | 69 / 1 / 10 | 56 / 1 / 23 |
| D | 10 / 3 / 35 | 11 / 1 / 34 | 9 / 1 / 36 | 8 / 1 / 37 |

The selected truck in A and bus in C remain matched while their surrounding frames lose matches under stronger darkening. In C, quarter-brightness reduces fusion TP from 72 to 56 despite the highlighted bus remaining detected. This directly demonstrates why a successful selected object cannot establish whole-frame robustness.

For B, half-brightness gains exactly GT index 26, the highlighted car, and loses no previously matched GT object at threshold 0.30. The FP count remains three; this does not assert identical FP identities. It remains below LiDAR's 21 true positives. The matching-details file retains assignments and FP boxes to support inspection.

## Visibility and LiDAR support

| Case | Target | Annotated visibility band | Current-scan LiDAR points |
|---|---|---|---:|
| A | truck | 80–100% | 73 |
| B | car | 80–100% | 2 |
| C | bus | 60–80% | 65 |
| D | traffic cone | 80–100% | 1 |

Visibility bands are original nuScenes annotation categories; they are not measured brightness, detection confidence or a per-pixel estimate in the chosen camera crop. Do not infer that B is severely occluded from this record: it has high annotated visibility but sparse keyframe LiDAR support. Nor does high annotated visibility guarantee a detection, as D illustrates. These observations do not identify a cause of failure. The model additionally uses nine preceding sweeps.

## Filtering and validation

Category mapping, strict class-dependent ego XY ranges, removal of GT boxes with zero combined LiDAR/radar support, and removal of bicycle/motorcycle centres within original bicycle-rack boxes follow the public nuScenes SDK filtering rules. Predictions have no zero-point rejection. Original rack geometry is now available. Source definitions: [SDK filtering](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/common/loaders.py), [category mapping](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/utils.py), [point-in-box geometry](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/geometry_utils.py).

The reconstructed filtered GT set matches the original preserved filtered GT in class, centre, size and quaternion to six decimal places for every A–D frame, with identical counts. The original GT ordering is retained for local diagnostic matching. Ten-class tables replace the earlier eight-class tables for whole-frame interpretation; the older outputs remain historical evidence. For B at 0.30 the counts happen to agree, but this equality must not be assumed for every case or threshold.

This is a transparent local implementation of the filtering and existing score-ordered diagnostic matcher, not a rerun of the official devkit AP evaluator or an independent cross-implementation validation. Box-centre containment uses inverse quaternion rotation and inclusive bounds. All-class matching uses score descending and preserved index for ties, class equality, unused GT and centre distance <2 m. Scores >= threshold are included.

## Night-case selection and remaining inputs

| Candidate | Token | Original scene description |
|---|---|---|
| scene-1059 | eb82a835419149aeae82c5e8ea64377b | Night, residential, car turning |
| scene-1060 | e0e5ebc17c73465181cab1383606e5f5 | Night, rain, turn, ped in dark |

Recovery found 15 night-labelled validation scenes. Candidate selection took the middle timestamped validation frame in each of the first two scene names sorted alphabetically, before inspecting predictions. This is deterministic convenience selection, not random or representative sampling. Both inventories refer to the same acquisition-log prefix; do not treat them as independent recording environments. Scene-1060 combines night and rain, so their effects cannot be separated.

Night image review and comparisons were subsequently completed in [night cases](../night_cases/README.md). The following inventory records the inputs required at the original metadata-recovery stage; it is not a current pending-work instruction. Only 12 camera keyframes, two LiDAR keyframes, their camera/LiDAR calibration and ego poses, and the two models' saved full-validation predictions for these tokens are needed for the next visual analysis. No sweep restoration or inference is needed if existing predictions are verified and extracted.

## Reproduce

From the parent case-analysis folder, with the restored source export and extracted six-case reference:

```bash
python metadata_completion/complete_checks.py --metadata case_metadata_and_night_candidates.json --reference reference --input input --output metadata_completion/results
```

The metadata file is retained separately in the evidence bundle. `ten_class_threshold_tradeoffs.csv` contains 48 frame/condition/threshold summaries; `focal_visibility.csv` contains four original annotation links; `completion_audit.json` records GT reconstruction checks and B's gained/lost GT indices. The script retains the existing nearest-object and threshold conventions and introduces no new fitted parameters.
