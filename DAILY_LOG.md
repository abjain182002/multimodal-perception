# Daily Project Log

## 29 October 2026 — real fog CPU pilot validated; GPU smoke prepared

- Verified uploaded 5,161,010-byte preflight ZIP SHA-256 9fd0f95619b8e4d19ef9a4c6432e332a90d75281c3a303159e76581bdb6a8126 and all 31 payload hashes. All 9,600 filenames are available, but full input-content verification remains separate. Six pilot cameras and ten raw scans remain private.
- Accelerated adapter matches pinned source exactly for 18 full-resolution RGB uint8 outputs and 30 raw float32 scan outputs. Three synthetic radius-boundary cases, deterministic repeats, identity, finite values, fifth-feature retention and lossless PNG round-trips pass.
- Recorded per-file RNG mapping and PNG encoding amendment before inference. No extra clipping of the source fog intensity/noise behaviour. Created a visually checked input-only camera/BEV composite; no detector outcomes shown.
- Four CPU helper tests pass, including path ambiguity handling and preservation of non-path annotation geometry in a synthetic 600-row-to-pilot cache preparation. GPU scripts compile but have not executed locally.
- Prepared isolated previously exercised PyTorch/CUDA stack restoration, original-source/pilot re-verification, strict two-checkpoint loading and fixed four-condition one-frame GPU smoke. Success/failure evidence is copied and hash-verified on Drive. Real GPU execution, full-study input validation, 600-frame fog inference and accuracy evaluation remain pending.

## 29 October 2026 — bounded fog extension started

- Reviewed pinned MultiCorrupt camera/LiDAR fog implementations and converters. Raw-scan coordinate processing and all historical sweeps are required; camera and LiDAR benchmark severities are not one calibrated physical visibility distance.
- Prepared a standard-library CPU preflight that fixes the original 600-frame population and first-token pilot rule, pins source/lookup-table Git hashes, inventories available sensor sources/runtime, and recovers a private six-camera/ten-LiDAR-file pilot where available.
- No prediction contents, GT outcome selection, package installation, GPU inference, fog corruption or source pickle execution in this stage. Exact implementation/RNG/encoding and any acceleration remain to be validated and frozen before full inference.
- Planned scope is two fresh unperturbed baselines plus six severity/model fog runs. Snow remains optional; broader extreme-weather discussion remains literature/limitations work.

## 28 October 2026 — CPU natural-weather analysis complete and imported

- Uploaded final ZIP: 2,156,223 bytes, SHA-256 75126afda2354f95f9402fccfa37100fb45d3f956cb1d5600cfb6fdf2c289227. All 69 manifest-listed payload sizes/hashes pass; uploaded bytes match the Colab-reported verified Drive copy.
- Both full replay checks pass: 32,451 numeric values each, maximum error 8.881784197001252e-16 at tolerance 1e-12. Local reconciliation repeats these recorded comparisons; no fresh local GT evaluation or inference.
- Four group ten-class mAP gains: +3.57/+6.26/+1.23/+0.93 points. Night-only and night/rain have 3/6 absent GT classes. Same-four-class AP and pedestrian results are separately reported; night/rain pedestrian AP 17.28/20.59 on 22 instances across frames. All night scenes share one log.
- Recomputed 400 AP values from saved PR curves; checked eight group rows, 80 class rows, NDS/mAP arithmetic, class-coverage sensitivity summaries and GT partition totals. Retained three class-specific regressions despite aggregate gains.
- Preserved complete unchanged source ZIP, readable CSV/audits, a visually checked three-panel coverage-aware figure and report discussion. New local figure/report are on GitHub; original Colab Drive receipt covers source outputs only. CPU stage complete; fog remains pending as a separate experiment.

## 28 October 2026 — real weather inventory reviewed; CPU evaluator ready

- Verified uploaded preparation ZIP SHA-256 d9aa5377247df83162cbfd8126638e4ded39e6459f036e378cf4f35cc47ad1a0 (184,365 bytes) and all five manifest-listed payloads. Reclassified all descriptions locally and checked all 6,019 sample memberships against 150 scene records.
- Reviewed all descriptions before subgroup metrics. Literal groups remain 111/24/12/3 scenes and 4,449/968/482/120 frames. There are 18 unique acquisition logs; all 15 night scenes share one log. No negated condition mentions; no automatic dry/daylight interpretation.
- Prepared CPU stage 2 with isolated Python 3.10.22, resolved dependency pins, original evaluator source hashes, verified metadata restoration and an evaluation-only sensor/pose derivative. Full annotation/velocity links remain intact. Each model must reproduce full summary and detailed metrics within 1e-12 before any subgroup scores are accepted.
- Pooled four-group evaluation reports per-class GT counts, absent/sparse classes, official ten-class scores and separately labelled class-coverage sensitivity AP. Descriptive point estimates only; no significance or weather-causality claims.
- Four SDK-backed local checks pass in the pinned environment, including synthetic original-versus-derived metadata evaluation, temporal velocity support, empty-frame/order preservation and report generation. Actual full-data subgroup evaluation and output Drive receipt remain pending user-run Colab execution; no GPU inference performed.

## 28 October 2026 — CPU weather-group preparation ready

- Added a metadata-only, standard-library preparation script for the full 6,019-sample / 150-scene population. Preserve protocol before grouping and do not open prediction contents in stage 1.
- Four disjoint strata use literal rain/night mentions; descriptions, negation flags and acquisition-log counts are retained for review. No automatic dry/daylight or causal-weather claims.
- Verify preserved validation token Git blobs and metadata table SHA-256 values; avoid full multi-GB ZIP rehash or image extraction. Produce manifest-verified ZIP and check Drive-copy hash.
- Three local synthetic checks pass, including full-size membership fixture, ZIP/TGZ and restored-metadata paths, incomplete-scene rejection, and invalid prediction bytes left unparsed. These are not real weather-group results.
- Real Colab execution, all-description review and stage-2 CPU evaluation remain pending. Fog is planned after natural-stratum analysis; snow optional, extreme conditions discussed as literature-supported limitations.

## 21 October 2026 — sourced cost/performance scenarios

- Recorded dated USD prices for OAK-1/PoE cameras, separately listed interfaces and a qualified Mid-360 retailer reference offer; retained backorder, tax/shipping and confirmation caveats. Excluded inaccessible DJI pricing, quote-only Ouster, an unreconciled sold-out high-price listing and historical launch prices from the current arithmetic.
- Generated 22 component scenarios for 11 sensor-count configurations and two camera product options. Added 162 explicitly assumed contingency/LiDAR-price sensitivity rows; unpriced compute, integration, power, tax and other items remain additive unknowns.
- Recomputed the preserved 600-frame accuracy-context differences (+3.8242 mAP / +1.8484 NDS points) and separate 150-frame three-session T4 latency/memory ratios (1.8935 / 3.6598). Neither measured point dominates the other on NDS versus latency; no global or financial frontier is inferred.
- Preserved standard-library reproduction code, inputs/source records, tables and a report discussion draft. Reused existing figure guidance to limit report space. No new inference, purchase, formal quotation request, hardware validation or original metric modification performed.

## 21 October 2026 — sensor configurations and bounded feasibility decisions

- Mapped 11 physical-input/availability scenarios, including one camera, one LiDAR, their combination, reduced camera availability, six-camera fusion, and two-/three-/multiple-LiDAR options.
- Corrected the scope inventory: one camera plus one LiDAR is already measured by KITTI CLOCs. Its mixed seed outcomes and training/source-recovery limits remain explicit.
- Inspected pinned ImVoxelNet and FCOS3D configurations and official BEVDepth code/model links. ImVoxelNet's checkpoint returned HTTP 200 and a 126,018,456-byte content length; no weight download, strict load or inference was performed. Verified that default nuScenes FCOS3D aggregates six physical views despite one-image forward passes.
- Verified Waymo's separate LiDAR suite and OpenPCDet's geometric merge, plus its explicit lack of distributed Waymo weights. Deferred a physical multi-LiDAR run until actual data/weights/resource gates pass; no universal claim that retraining or a new encoder is mandatory.
- Preserved the source-backed assessment, 11-row configuration CSV, decision/source records and a compact report table/discussion draft. Preferred next camera-only step is a bounded KITTI ImVoxelNet preflight, conditional on data and model checks. Supervisor acceptance and monetary sourcing remain pending; no new GPU experiment or original-metric change.


This file records work chronologically. `PROJECT_HISTORY.md` remains the synthesized master dissertation/technical record; this log preserves session-level details so that nothing important is lost between larger consolidations.

---

## Falcon + LiDAR single-frame fusion

### Objective
Integrate Falcon Perception with the existing KITTI LiDAR projection and the already frozen median-ROI depth association method on one frame before expanding to a multi-frame experiment.

### Controlled variables
The LiDAR side was intentionally kept unchanged from the YOLO baseline:

- KITTI calibration matrices (`P2`, `R0_rect`, `Tr_velo_to_cam`),
- projected LiDAR points in camera coordinates,
- points associated when their image projection lies inside a 2D detector box,
- minimum of 3 projected LiDAR returns,
- depth estimate = median camera-coordinate LiDAR depth inside the box.

### Changed variable
The camera detector input was changed from YOLOv8x to **Falcon Perception + score-free deduplication at IoU 0.50**.

### Dataset / frame
KITTI training frame:

```text
000149
```

### Environment

```text
~/Major_Project/envs/falcon310
Python 3.10.21
```

### Implementation
Created:

```text
scripts/fuse_falcon_lidar.py
```

The script reuses `adapters/falcon_adapter.py`, converts Falcon detections to project box format, applies the already validated score-free deduplication rule, projects KITTI LiDAR into the camera frame, associates points with each deduplicated box, and estimates median depth.

### Problem encountered
The first runtime attempt failed with:

```text
ModuleNotFoundError: No module named 'matplotlib'
```

### Diagnosis
`matplotlib` had not yet been installed in the dedicated `falcon310` environment. The script itself had already passed `py_compile`, so this was an environment dependency issue rather than a Python syntax problem.

### Fix
Installed Matplotlib into `falcon310`:

```text
matplotlib 3.10.9
```

Import verification succeeded, after which the fusion script ran end-to-end.

### Result
Falcon output on frame `000149`:

```text
Raw Falcon boxes: 11
Deduplicated boxes: 10
Fused detections: 10
```

All 10 deduplicated boxes had at least 3 associated projected LiDAR points.

Median LiDAR depth estimates:

| Detection | Estimated depth | LiDAR points |
|---:|---:|---:|
| 0 | 59.97 m | 7 |
| 1 | 49.60 m | 21 |
| 2 | 41.58 m | 31 |
| 3 | 47.37 m | 66 |
| 4 | 47.44 m | 102 |
| 5 | 27.71 m | 92 |
| 6 | 27.85 m | 129 |
| 7 | 15.73 m | 309 |
| 8 | 13.21 m | 767 |
| 9 | 9.22 m | 1,629 |

### Qualitative verification
Generated locally:

```text
~/Major_Project/model_outputs/falcon_lidar_fusion/falcon_lidar_000149.png
```

The image was visually inspected. Projected LiDAR returns were aligned with the detected vehicle regions, and the near-to-far ordering of the estimated depths was physically plausible.

A presentation limitation was noted: dense LiDAR points and overlapping labels make the current visualization cluttered. This is a visualization issue, not evidence of a fusion failure, and can be cleaned later for dissertation presentation.

### Git synchronization issue
The first push of the new local commit was rejected because the remote `main` branch already contained newer documentation commits (`PROJECT_HISTORY.md` and the README link).

The repository was synchronized safely with:

```text
git pull --rebase origin main
git push
```

No force-push was used. The rebase succeeded without conflict and the final working tree was clean.

### Final commit

```text
bf3154d — Add Falcon LiDAR single-frame fusion
```

### Dissertation-safe interpretation
This experiment demonstrates successful **single-frame software integration** of Falcon detections with the frozen KITTI projected-LiDAR median-depth association pipeline. It is a qualitative/engineering validation step, not yet a multi-frame accuracy result and not evidence that Falcon+LiDAR outperforms YOLO+LiDAR.

### Next step

- multi-frame Falcon + LiDAR evaluation,
- then controlled YOLO+LiDAR vs Falcon+LiDAR comparison using matched evaluation conditions.

### Progress checkpoint
Working technical-completion estimate after this milestone: **approximately 51%**.

## 2026-09-19 — Controlled YOLOv8x + LiDAR vs Falcon + LiDAR

### Objective
Complete the controlled multimodal comparison while keeping the downstream LiDAR association fixed.

### Environment
- YOLOv8x evaluation: `azitt310`
- Falcon evaluation: `falcon310`
- Apple MPS used for YOLOv8x inference

### Fixed evaluation settings
- same 10-frame development split and 10-frame independent holdout split
- KITTI `Car`
- match IoU = 0.30
- minimum projected LiDAR points = 3
- same calibration/projection
- same V1 median-ROI depth estimator
- YOLOv8x confidence = 0.25
- Falcon deduplication IoU = 0.50

### Verified results
Development:
- YOLOv8x + LiDAR: 45 matched fused cars, MAE 1.492 m, Median AE 0.923 m, RMSE 2.402 m, signed error -0.860 m
- Falcon + LiDAR: 41 matched fused cars, MAE 1.783 m, Median AE 1.105 m, RMSE 2.872 m, signed error -1.261 m

Holdout:
- YOLOv8x + LiDAR: 41 matched fused cars, MAE 1.789 m, Median AE 1.471 m, RMSE 2.355 m, signed error -1.029 m
- Falcon + LiDAR: 42 matched fused cars, MAE 1.655 m, Median AE 1.477 m, RMSE 2.316 m, signed error -0.948 m

### Validation
Both YOLOv8x development and holdout CSVs were independently re-read and the aggregate metrics were reproduced.

### Issues encountered
- `falcon310` did not contain Ultralytics, confirming the need to keep model environments separated.
- the first converted YOLO evaluator failed with `NameError: name 'iou' is not defined`; the shared IoU helper was restored and the full run then completed successfully.

### Interpretation
YOLOv8x was stronger on development, but the ordering did not transfer cleanly to holdout. Falcon produced one extra matched fused car and slightly lower holdout MAE/RMSE, while Median AE was almost identical. Treat this as a split-dependent tradeoff rather than a universal detector ranking.

### Artifacts
- `results/kitti/camera/lidar_depth_comparison/yolov8_falcon_lidar_comparison.csv`
- `results/kitti/camera/lidar_depth_comparison/figures/yolov8_falcon_lidar_comparison.png`
- `scripts/evaluate_yolov8_lidar_multiframe.py`
- `scripts/evaluate_yolov8_lidar_holdout.py`

### Next step
Finish repository documentation and checkpoint this experiment before moving to the next technical stage.

## 2026-09-25 — KITTI CLOCs GT audit and candidate-IoU preparation

- Mac `azitt310`: 84 full-suite tests passed after the GT audit script was pulled.
- Canonical train/validation split: 3,712/3,769 frames, no overlap. Audited all 7,481 labelled KITTI frames; exact Car GT 28,742 before and 28,374 after detector range filtering. Train/validation retained 14,144/14,230 Cars. Saved `results/kitti/clocs/training_exports/kitti_clocs_gt_audit.json` at `96c3a37` and confirmed a clean Mac tree.
- CPU reference implementation now computes camera-space rotated 3D IoU for dense candidate matching. PointPillars extraction returns matching camera boxes in candidate order. Seven new synthetic tests pass; locally available suite 43/43.
- Gate: `reference_structural_smoke` has placeholder zero overlaps. Export genuine dense 3D boxes from Colab and independently verify real targets before training or declaring fusion accuracy.

- Added guarded per-frame target builder and GPU export contract. A synthetic end-to-end test exercises split verification, target generation, saved SHA-256 and reload; local subset 49/49 tests passed. Genuine dense candidate export and real-data checks remain the GPU gate.

- Mac confirmation: pulled `e2e3a47`; 97/97 repository tests passed in 0.613 s, with clean `main...origin/main`.
- Tightened CLOCs target input contract: regenerate all rows from saved original 2D detections and projected dense candidate boxes; reject altered 2D geometry, scores, indices or row types. Local subset 52/52 tests passes. Real candidate labels remain pending GPU export.

- Mac confirmation: pulled `6b913a1`; **100/100 tests passed in 0.551 s**, and `main...origin/main` was clean.
- Set daily Phase 1 targets through 10 October in `docs/PHASE1_DAILY_TARGETS_2026-09-25_TO_10-10.md`. Each date has an evidence gate and a GitHub checkpoint or documented blocker. Dissertation chapter drafting starts after 10 October; a continuing GPU outage is escalated rather than recorded as finished work.

- User reaffirmed the locked 29 August scope. Updated `docs/PHASE1_DAILY_TARGETS_2026-09-25_TO_10-10.md` to track the Azitt fusion baseline, OpenPCDet/PV-RCNN++, MultiCorrupt, Glasgow demo and NVIDIA optimisation (stretch only) explicitly, while preserving all already verified work. Added inventory and provenance rules for existing and future result figures.

## 2026-09-26 — Real CLOCs train frame with positive targets

- Pinned PointPillars GPU export and Mac SHA-256 verification produced 107,136 dense candidates for canonical SECOND train frame `000149`, using the official reduced cloud (`119,984` raw points to `19,722` reduced; candidate SHA-256 `3943f256e48e1d9b072cf01f8a9165c3cff9f612b94125bd1986cf21668b1b99`). The export manifest records checkpoint, config, split and provenance.
- On Mac, frozen YOLOv8x produced 9 Car detections. After the detector range filter, KITTI labels retained 10 Car GT. The regenerated CLOCs features have 158,838 rows; the 3D IoU audit has 5,137 nonzero candidate overlaps, maximum 0.9291334748, and 179 positive / 106,798 negative / 159 ignored candidates (sum 107,136). Saved `results/kitti/clocs/training_targets/frame_audits/clocs_real_frame_000149_audit.json`.
- `000000` is a legitimate train-frame negative control: zero retained Car GT and zero YOLO Car detections, with 107,136 negatives. Its audit is already preserved in `results/kitti/clocs/training_targets/frame_audits/clocs_real_frame_000000_audit.json`.
- Both frames are single-frame pipeline integrity evidence; no fusion model has yet been trained or evaluated for KITTI AP. The large `.npz` examples remain outside Git on Mac, with the 3D candidate export backed up to Drive. Next daily gate is the planned multi-frame training preparation.

- Geometry reference gate: pinned MMDetection3D KITTI CUDA evaluator exited with signal 11 before writing a comparison; optional Lyft import was bypassed separately. Independent SciPy/Qhull CPU comparison of hash-verified train frame `000149` **passed** on Mac (`azitt310`, SciPy 1.14.1): 258 sampled anchors (65 positive, 65 ignored, 64 overlapping negative, 64 zero IoU), maximum absolute IoU error 2.968672907499581e-08 versus 0.0005 tolerance, zero threshold/label disagreements, and full-frame counts verified. The complete Mac test suite passed 108/108. The measured JSON report and parity PNG were saved in `results/` at `5214842`. This is a single-frame CPU geometry check, not a passed pinned GPU evaluator or KITTI AP.
- Colab reported GPU access denied due to usage limits at approximately 11:26 IST. The Mac geometry audit and figure are preserved; no additional GPU results are claimed. The native CUDA evaluator crash and multi-frame GPU export remain open blockers. Resume the planned multi-frame preparation when compute is available; keep validation frozen until evaluation.

- Train-only GPU input pilot: Mac `azitt310` scanned all 3,712 frozen SECOND training frames; the four Car-GT strata contain 473 zero / 1,208 one-to-two / 1,328 three-to-six / 703 seven-plus frames, matching the saved GT audit's zero-frame count. Deterministic 16-frame selection includes `000000` and `000149` and spans 5 zero, 4 low, 3 medium and 4 crowded frames (67 retained Car GT total). The manifest records the exact selected-ID SHA-256 `17b02589a45ec5046829854fe6c533236596a10dd9ede3c6205f2cb0df38d263`, split hashes, pinned checkpoint identity and SHA-256 of each selected image, raw cloud, label and calibration. Mac full suite passed **109/109**; `results/kitti/clocs/training_exports/clocs_train_export_batch_manifest.json` was pushed at `c3d58eb`. **Export is pending:** the deliberately balanced pilot is not representative of the full training distribution, and no new candidate predictions or fusion scores were produced.

- Mac offline preflight and archive: `azitt310` passed **110/110** tests. Five pinned external repository revisions matched the inventory; the Azitt PV-RCNN++ checkpoint was absent at the path checked, and nuScenes metadata was absent at the three checked locations. Saved `results/audits/environment/phase1_mac_dependency_inventory.json` at `2f53c9e`; path presence alone does not verify model inference.
- Packaged the exact 16 frozen training frames (64 KITTI inputs plus train/validation splits) into a locally verified **30,768,829-byte** archive at `$HOME/Desktop/clocs_train_export_16.tar.gz` (SHA-256 `3600369f1d513bb48b0896200486cb9925c4f1e3a7e50a960f2858677b59e1d2`). The embedded manifest SHA-256 is `71f4f933c9a2b1228b598ce604e9016f3ea295437845a94ec182bd0407eaf054`. Recorded the user-provided Mac packaging output in `results/kitti/clocs/training_exports/clocs_train_export_16_archive_receipt.json`; the archive remains outside Git. **Colab transfer, restore verification and multi-frame GPU exports remain pending.**

- Colab GPU access resumed on a fresh Tesla T4 runtime. User uploaded the exact **30,768,829-byte** 16-frame input archive; SHA-256 matched the Mac receipt, and a second Python standard-library pass validated the embedded manifest SHA-256 and all 64 KITTI input-file and two split-file hashes. Extracted to `/content/datasets/KITTI/training` with splits under `/content/datasets/KITTI/splits`; `RESTORE PASS: 16 frames, 64 KITTI files, 2 split files`. Drive backup SHA-256 was verified. Saved the user-reported restore audit at `results/kitti/clocs/training_exports/clocs_train_export_16_colab_restore_audit.json`. Fresh runtime still lacked `/content/pointpillars310`, pinned MMDetection3D, the checkpoint and the private project checkout; **GPU candidate export remains pending**.

## 28–29 September 2026 — CLOCs training and baseline reproduction

- Verified export and target construction for 128 train-only KITTI frames. CLOCs optimization used 96 frames and internal development used 32; the full 128 were not all optimizer training data.
- Compared all nonignored candidates (ratio0) with ratio8 negative sampling on one seed. Internal development candidate AP was 0.85995 versus 0.74268. This is a setting-specific ranking result, not KITTI detection AP or a general rejection of sampling.
- Completed full 3,769-frame fusion validation. Investigated a historical PointPillars mismatch using saved-prediction replay, raw versus camera-FOV-reduced cloud probes and KITTI range filtering. The corrected reduced-FOV run reproduced all nine historical PointPillars AP40 metrics exactly.
- Baseline-matched strict Car 3D AP40: PointPillars 90.3449 / 79.1299 / 76.1566; CLOCs 90.1409 / 79.2542 / 73.6326 (Easy / Moderate / Hard). Mixed effects: +0.1243 Moderate and -2.5239 Hard AP points. The earlier raw-cloud paired result remains a separate diagnostic.
- Coverage diagnostic across all labelled Cars: 10,650 baseline versus 10,447 fusion covered at best 3D IoU >= 0.70; 186 rescued and 389 lost. This is not official difficulty/ignore matching or one-to-one recall. Evidence is under `results/kitti/clocs/baseline_matched_validation/`.

## 30 September 2026 — PV-RCNN reproduction and runtime

- Updated stale project summaries and generated verified baseline-matched CLOCs AP40/coverage PNG/PDF figures, source CSVs and captions.
- Reused official OpenPCDet commit `233f849829b6ac19afb8af8837a0246890908755` and its KITTI PV-RCNN model-zoo checkpoint (SHA-256 `08552932026855bcc10f66dbc45409f139d3208fd0095c2aab1f459f055fe463`). This is PV-RCNN, a two-stage detector, not PV-RCNN++ or Azitt.
- Built a separate Python 3.10/Torch 2.1.2+cu121/spconv-cu120 2.3.6 environment on Kaggle T4. Corrected source-tree shadowing, restored Python helper files omitted from the wheel from the same pinned checkout, and pinned Kornia 0.6.7 for the imported helper API. All 367/367 checkpoint parameters loaded; four-frame finite-output smoke passed.
- Ran all 3,769 frozen validation frames using official KITTI preprocessing/conversion. Car strict IoU 0.70 3D AP40 = 92.1047 / 84.3604 / 82.4830. Native OpenPCDet and pinned MMDetection3D evaluation matched exactly across all nine metrics. Prediction SHA-256 `80dbeef521b289607f635df07062b4c9171576491709731b63351eaf03dd7381`.
- Verified downloaded full evidence archive SHA-256 `709f818e3528acbb41e47cbb59fb0d541a188ac324d0269b7dd02ded017813a1`; compact results, comparison PNG/PDF/CSV and findings are in `results/kitti/pv_rcnn/validation/`. The comparison is between pipelines with different architectures and training, not an equal-budget ablation.
- Completed synchronized batch-one runtime measurements: 10 warm-up calls, 32 fixed validation frames, three passes (96 calls), one Tesla T4. Pipeline median 176.31 ms, p95 196.16 ms; forward/postprocessing median 149.45 ms, p95 168.05 ms. Serial measured rate 5.54 calls/s. Excludes loading, annotation conversion and writing; repeated reads can be cached.
- Recomputed statistics from raw CSV and verified executed source/ZIP hash (`23bd999fa5c64cc825d6780e6a5c4322cc701d55ae2903e21d2dac16c924efd7`). Runtime evidence and SVG figure were pushed at `1e695b1`, under `results/kitti/pv_rcnn/runtime/`. Runtime comparisons with other models remain pending.
- Next priority after feedback: audit checkpoint-specific training-split provenance and camera pilot overlap; test CLOCs training adequacy with aligned preprocessing, a frozen internal dev split, multiple seeds and a larger train-only subset. Validation has already been inspected and must not be treated as untouched final test data. No new CLOCs training or semantic-fusion result is claimed.

## 30 September 2026 — Aligned CLOCs training study

- Re-exported 128 training frames with camera-FOV-reduced clouds and verified dense candidates, feature rows and targets. Fixed 96 optimization / 32 development frames across three initialization seeds; all valid candidates used, 12 epochs, Adam LR 0.002.
- Development candidate AP: 0.877935 / 0.866960 / 0.866751. Selected seed 20260928 using development AP before its full-validation run. Candidate AP is not KITTI AP40.
- Selected seed achieved Moderate 3D AP40 79.8616 against paired PointPillars 79.1299; Hard decreased to 74.8719. Full baseline parity error 0.0. Reports, learning curves and comparison figures preserved at `0b8674a` under `results/kitti/clocs/aligned_seed_study/`.

## 1 October 2026 — Full-validation seed variability and epoch extension

- Full re-inference and AP40 evaluation for seeds 20260929 and 20260930 completed on all 3,769 frames. Combined epoch-12 Moderate 3D AP40: 79.8616 / 76.2666 / 78.7616, versus PointPillars 79.1299. Only one seed improves Moderate; all three reduce Hard. Three runs support descriptive consistency analysis, not significance testing.
- Verified 516 saved bulk-training files against their manifest. Restored seed 20260928 epoch-12 model and Adam state; development AP reload error 0.0. Continued epochs 13–24 at constant LR 0.002 with no scheduler, preserving the original 96/32 split. Original checkpoint lacks RNG states; continuation is not claimed to be bitwise uninterrupted reproduction.
- Development candidate AP increased from 0.877935 to 0.883568; best epoch remained the final epoch 24, so plateau remains unestablished. Full epoch-24 3D AP40: 90.2610 / 79.4874 / 74.7797. Moderate fell 0.3741 points versus epoch 12. Candidate ranking improvement did not translate into improved 3D detection AP40 in this diagnostic.
- All new full-validation evaluations reproduce all nine historical PointPillars AP40 metrics exactly. Validation is iterative, not an untouched final test. Results concern this small-data implementation; they do not establish general CLOCs or fusion failure.
- Mac independently verified and restored both full evidence archives. Three-seed ZIP SHA-256: `05e9312e6d3d2e0e91340b6f74395707cfc7e40b9f2f32536768cea8a79f12f2`; epoch-24 full-validation ZIP: `68d419a06222d8f22b40483cddedae78e37a84e1a4effcd7f1d9d2a85d44a559`.
- Committed 17 compact result/source files, exact comparison CSV, two SVG figures, receipts and findings at `66584df`. Figure generation completed; final visual inspection remains required before dissertation use. Evidence: `results/kitti/clocs/seed_diagnostics/`.
- Frozen this diagnostic stage; larger-frame adequacy remains untested. Next integration requires an official source/checkpoint and data feasibility gate, with reuse of existing repositories.


## 2 October 2026 — Paired BEVFusion nuScenes subset evaluation

- Restored hash-verified parts 01–02 sensor backup and froze 1,231 samples from 31 complete official validation scenes (20.45% coverage). Availability-selected, not random; part 03 excluded.
- Converted metadata with up to nine preceding LiDAR sweeps. Preserved full SDK lookup tables while restricting conversion iteration; corrected dataset-token checks for serialized MMEngine data.
- Verified CUDA operations and strict loading of both official pretrained checkpoints. Generated and backed up all predictions. Fixed explicit CPU conversion before official result formatting; reused saved LiDAR predictions.
- Official custom-split evaluation confirmed exact prediction/GT token equality for each pipeline. LiDAR mAP/NDS = 0.6421140063400634 / 0.6920865595499972; camera–LiDAR = 0.6781470812751105 / 0.7115433802870929. Differences = +3.6033 mAP points / +1.9457 NDS points. Filtered GT has 32,276 boxes across all ten classes.
- Metrics and official curves were verified on Drive. Added a compact summary transcribed from execution output, comparison CSV, two figures in SVG/PNG/PDF, plotting script and figure hashes. Raw evaluation files have not yet been imported into this commit; their reported hashes and Drive location are recorded.
- Updated stale README status to distinguish completed mini/subset measurements from pending full validation. No extrapolation or controlled camera-ablation claim is made.


## 7 October 2026 — Original nuScenes subset evidence import

- Opened the supplied 65,968,262-byte evidence ZIP; SHA-256 `7d7db1743cabb1b905baf3feefb46b63a510bbd532a04645b2b3db33ad60ab6e` matched the receipt. All 84 members matched manifest sizes and hashes.
- Both prediction token sets equal the frozen 1,231 samples. Prediction and metric-summary hashes match original evaluation audits; stored mAP/NDS arithmetic matches stored class APs and TP scores within 1e-12.
- Imported compact original metric summaries, scripts/configurations, audits, frozen tokens/custom split, environment record and both official summary PDFs. Preserved a complete source-member manifest and verification receipt; large predictions and annotation data remain in the verified archive.
- This verifies saved evidence, not recomputed accuracy from ground truth. CPU evaluation replay and full-validation expansion remain pending.



## 7 October 2026 — Repository organization

- Physically grouped 411 result files under KITTI, nuScenes and shared audits. Related camera/depth studies, PointPillars cases, CLOCs pilots and final diagnostics now have separate folders.
- Added experiment READMEs, a figure gallery, scripts guide and complete path migration map. Shortened the root README and retained its detailed content in docs/PROJECT_GUIDE.md.
- Updated active script defaults and documentation references. Frozen metrics, figures, original source and provenance manifests retain their original bytes. This is an organization change, not new experimental evidence.

- Follow-up preservation audit: every one of the 550 baseline repository files is present at its mapped path; all 404 non-Markdown result blobs match the pre-organization originals. Restored seven archived TXT/Python records to their original blobs after detecting path-only rewrites. Saved the repository preservation audit under results/audits/repository/.


## 8 October 2026 — nuScenes subset ground-truth replay

- Restored hash-verified full metadata and the frozen 31-scene custom split. Created a CPU Python 3.10.22 evaluator environment; all four recorded nuScenes-devkit source hashes matched.
- Recomputed metrics from the unchanged paired predictions and ground truth for exactly 1,231 frozen tokens. Filtered GT counts match the original evaluation.
- Both headline mAP/NDS values reproduced exactly. Maximum summary/detail absolute error was 8.881784197001252e-16, below the 1e-12 tolerance; wall-clock evaluation time was excluded.
- Verified the supplied 1,161,924-byte replay ZIP SHA-256 aa421f2e9fb092fabf56a004f26a54da1d6a7fdd755631ab22bbe33674728726, ZIP CRC, all 60 manifest member hashes/sizes and consistency of scripts/audits/stored metrics.
- Imported compact replay records under results/nuscenes/bevfusion/validation_subset/replay_20261008/. Complete detailed metric arrays and PR/TP PDFs remain in the verified Drive ZIP; its complete member manifest is committed.
- Same-evaluator reproduction is now verified. Full-validation expansion, subset representativeness, and mini/subset overlap remain unresolved; no additional ground-truth evaluation was performed during this import.


## 8 October 2026 — nuScenes parts 03–06 staging and backup coverage

- Reused the hash-verified part-03 keyframe backup; selectively extracted and verified required validation sensor files from the remaining part-03 LiDAR and parts 04–06 keyframe/LiDAR archives. All eight backups have verified Drive receipts. Temporary downloaded archives and duplicate local backup ZIPs were removed after verification; sensor files and Drive backups were retained.
- The union of parts 01–06 backup indexes covers 51,151 / 94,180 required files and complete inputs for 3,273 / 6,019 samples (54.3778%), across 82 complete scenes, with zero partial scenes. All original 1,231 frozen tokens remain covered.
- Coverage verification uses ZIP indexes, sizes and previously verified receipts; it does not repeat all sensor-content or full archive hashes. New inference/accuracy evaluation remains pending.
- Verified the supplied 1,998,546-byte staging evidence ZIP SHA-256 df02691f4838549e020c7deda00bfd5342334c5702ae652eefd9c9ced8e9a108, CRC, all 12 source-member hashes/sizes, coverage totals and eight receipt/archive provenance matches.
- Imported compact audit, scene/sample records, receipt summaries and complete source-member manifest under results/nuscenes/preparation/parts01_06_20261008/. Full sensor-member maps and available-file list remain in the verified Drive evidence ZIP; no measured accuracy records were changed.


## 9 October 2026 — Import complete nuScenes validation input staging evidence

- Verified the supplied 4,072,624-byte ZIP SHA-256 78f6ba980a6ab52e62087f48006e08395eda77a448cbadc8693a46fbc6b8b0f3, CRC and all 21 source-member hashes/sizes.
- The original 8 October backup-index audit covers 94,180 / 94,180 required files, 6,019 unique validation sample tokens and 150 complete scenes, with zero missing required inputs. Receipt archive identities and scene/sample totals agree. Both part-07 archives had zero required-file matches; their audits are preserved.
- Imported compact coverage records, receipt summaries, zero-match audits and full source-member manifest under results/nuscenes/preparation/parts01_10_20261008/. Full receipt sensor-member maps remain in the supplied source ZIP and Drive backups.
- This import verifies saved staging records, not current sensor contents or fresh archive hashes. Full validation inference and accuracy evaluation remain pending. Previous measured metrics and historical staging records remain unchanged.


## 9–10 October 2026 — Full nuScenes validation and original evidence import

- Restored and hash-verified the required sensors in three scene-complete storage batches: 2,368 / 2,368 / 1,283 samples, covering 150 scenes. Rebuilt pinned Python 3.10 / Torch 2.1.2+cu121 / MMDetection3D fe25f7a5 with CUDA 12.1 and GCC 11; synthetic BEV pooling and voxelization tests passed.
- Completed paired official pretrained LiDAR / camera–LiDAR inference. Verified all six batch prediction hashes and exact tokens, consistent model/configuration/seed, duplicate-free merge and equality with the 6,019 official validation sample tokens. Global evaluation followed merging; batch metrics were not averaged.
- Full mAP ×100 = 64.4301 / 68.3099; NDS ×100 = 69.2353 / 71.1352. Gains +3.8798 / +1.8998 points. Class AP rises for all ten classes; scale, orientation and attribute TP errors slightly worsen.
- Verified supplied 2,118,593-byte original evidence ZIP SHA-256 37a5882333283e882d21d8f165b308f79a5039a8e1064334ad480b93bd665269 and all 111 member hashes/sizes. Stored mAP/NDS arithmetic agrees within 1e-12; this import does not replay GT evaluation. Raw merged predictions remain separately on Drive with recorded hashes.
- Imported all original compact evidence bytes, four visually reviewed figures in PNG/PDF/SVG, exact CSV tables and plotting source under results/nuscenes/bevfusion/full_validation/. Updated current status indexes while preserving prior subset/mini and historical logs.
- Full accuracy evaluation is complete; separate full-split replay, selected nuScenes case studies, comparable runtime, robustness and open CLOCs analyses remain pending. This compares pretrained pipelines, not a controlled camera ablation or project training.


## 10 October 2026 — Source-checked supervisor benchmark tables

- Added `results/benchmarks/BENCHMARK_COMPARISONS.md`, two numerical CSVs, source URLs/text hashes and a compact manifest.
- Compared full nuScenes measurements with the same checkpoint-linked MMDetection3D reference: mAP/NDS deficits approximately 0.47/0.36 points (LiDAR) and 0.29/0.26 points (camera–LiDAR). Cause unresolved; rounded upstream values are not exact replay targets.
- Preserved all three aligned CLOCs seeds and epoch-24 continuation in the measured KITTI AP40 table. OpenPCDet AP11 and MMDetection3D unspecified AP are separate context, with no direct AP40 deltas.
- No new inference, ground-truth replay, case-study rendering, training or deletion performed. Case studies and outstanding diagnostics remain next work.


## 2026-10-14 — preserve case studies prepared on 13 October

- Preserved six annotated nuScenes figures, captions, exact object records, plotting code, input archive hash and output manifest under [case studies](results/nuscenes/bevfusion/case_studies/20261013/README.md).
- Verified all output manifest hashes and Git blob hashes before publishing.
- Added a visual case index and navigation links. Existing result files retained.
- Diagnostic operating point: confidence >=0.30, same-class XY centre distance <2 m. Selected examples are not population estimates or controlled camera ablations; visible conditions are not proven causes.
- Next: freeze a separate robustness protocol and sample set; full-validation replay and remaining CLOCs diagnostics remain pending.


## 2026-10-14 — camera-availability subset evaluation and preservation

- Froze 600 samples at four timestamp positions in each of 150 official validation scenes, independent of prediction outcomes. Restored and verified 9,600 sensors (4.70 GB).
- Restored pinned core framework/source/checkpoint versions. Current nvcc 12.1.66 differs from previous 12.1.105; compiler, GPU and synthetic custom operations checks passed, and environment records are retained.
- Strict checkpoint loads and one-sample camera-mask checks passed; inferred 600 frames for each of four conditions and backed up outputs every 50 frames.
- Evaluated identical 600-token GT sets using nuScenes-devkit with restricted GT sample iteration. mAP/NDS (%): LiDAR 64.31/69.44; intact fusion 68.13/71.29; front/back retained 63.63/68.76; front missing 64.84/69.44.
- Imported all 160 compact archive members with archive SHA-256 and all 158 manifest hashes verified. Recomputed stored mAP/NDS arithmetic within 1e-12 and reviewed three figures, saved in PNG/PDF/SVG with CSV tables and plotting code.
- [Experiment folder](results/nuscenes/bevfusion/camera_availability/20261014/README.md) distinguishes missing-view robustness from purpose-trained sensor configurations and computational-cost claims. Raw predictions and replaceable sensor backups remain on Drive. Controlled runtime/cost analysis, accuracy replay and remaining diagnostics are still open.



## 2026-10-15 — runtime benchmark, architecture and initial cost sources

- Restored pinned Python/PyTorch/framework/source/checkpoints and compiled CUDA 12.1.66 operations; GPU checks passed.
- Froze 150 samples (second selected timestamp position per scene from the prior 600-frame study); restored and individually hash-verified 2,400 sensors, 1.17 GB. Prior backup sizes/receipt hashes checked; full archives not rehashed in this restoration.
- Completed six isolated sessions (L/F, F/L, L/F), 20 untimed warm-ups and 150 timed frames/session. Three passes/model give 900 measured rows. Primary test_step timing excludes file loading; TF32/autocast disabled, single CPU thread.
- Median-of-session-medians latency: 287.81/544.98 ms; peak allocated memory: 609.82/2,231.80 MiB for LiDAR/fusion. End-session SM-clock snapshots vary; shared-cloud and temperature limits recorded.
- Verified original ZIP SHA-256 and all 52 manifest-member hashes (53 members including manifest). Reproduced per-session latency/loading statistics and paired token/point-count checks. Generated three runtime figures and two architecture diagrams, in PNG/PDF/SVG, with CSV tables and reproducible scripts.
- [Runtime evidence](results/nuscenes/bevfusion/runtime/20261015/README.md), [architecture](docs/architecture/README.md), [initial cost framework](results/cost_performance/README.md). Current matched monetary sensor costs, additional challenge conditions and remaining evidence diagnostics are still open.


## 15 October 2026 — controlled image-darkening challenge

- Frozen the same 600 frames / 150 validation scenes used for camera-availability analysis; fresh clean baseline plus image-value gains 0.50 and 0.25.
- Applied float32 multiplication to six resized camera tensors before normalization; files, LiDAR inputs, calibration and checkpoint unchanged. Autocast/TF32 disabled; no training.
- Preserved two failed smoke checks caused by metadata-comparison serialization, then successful corrected three-condition checks; failures occurred before prediction.
- Completed and Drive-verified 1,800 predictions and same-token nuScenes-devkit 1.2.0 evaluations.
- Clean / 0.50 / 0.25 mAP ×100: 68.1300 / 67.2352 / 64.5150; NDS ×100: 71.2886 / 70.8179 / 69.1982.
- Verified the 2,491,849-byte evidence ZIP SHA-256 `0d0a5ddd05185e856b9704df84c2a109af9e8cf6a16a44f2a99ad7b2aa9dd6b3`, all 120 members, and stored metric arithmetic; generated three figures in PNG/PDF/SVG and three CSV tables.
- Preserved original evidence and navigation links. Raw predictions and sensor backups remain on Drive. Synthetic darkening is not a real-night test; no confidence intervals or runtime claim.
- Remaining supervisor work includes other controlled robustness challenges, state-of-the-art/failure-handling synthesis, cost analysis and camera-only/multiple-LiDAR feasibility.


## 15 October 2026 — matched LiDAR point-reduction study

- Froze six conditions on the same 600 frames / 150 scenes: fresh LiDAR-only and fusion baselines, 50% and 25% retained points.
- Deterministic token-seeded PCG64 permutation yields nested subsets with original row order restored; original point attributes unchanged. The six-condition one-frame smoke check passed.
- Completed 3,600 predictions, global formatting and six same-token evaluations. Per-frame records confirm identical original/retained points across both models and unchanged image/metadata hashes across retention levels.
- At 25% retention, LiDAR-only/fusion mAP ×100 = 51.0381 / 60.2341; losses from own baselines = −13.2695 / −7.8961 points. Fusion retains an advantage but also degrades.
- Verified ZIP SHA-256 d5086b07200dd4a47668ca95e488129d0d605ce77d8f82fec9fad3a62ad51728, 4,498,505 bytes, all 218 members; recomputed stored metric arithmetic and per-token selection-index hashes for all 600 frames.
- Preserved original evidence, resume checkpoint, compact two-panel accuracy figure, supporting class heatmap and four CSV tables. Raw predictions/sensors remain on Drive.
- Recorded the 40–50-page report target and combined-figure policy; blur, noise, calibration, critical literature and monetary cost analysis remain planned. Reconciled outdated case-study/runtime status in guide/findings without changing original metric evidence.



## 16 October 2026 — blur, noise and calibration challenges

- Froze seven conditions on the existing 600 frames / 150 scenes; CPU transformations and 10,800 calibration checks passed.
- Restored single-T4 Kaggle inference; retained dependency and sweep-path repairs before successful seven-condition model smoke.
- Completed 4,200 predictions; input ledgers, raw export hashes and Drive preservation verified.
- Evaluated all seven against identical 12,318 filtered GT objects. Clean mAP/NDS ×100 68.1018/71.2767; strongest tested blur/noise/yaw mAP losses 2.2198/4.9663/3.4680 points.
- Imported all 270 original compact evidence members unchanged; rechecked archive/member hashes, stored metric arithmetic and 4,200 input records. No fresh GT replay or raw-prediction rehash on import.
- Produced one three-panel main-report figure, a supporting class heatmap, vector exports and three CSV tables with a reproducible script.
- Planned corruption studies complete; literature/failure synthesis, monetary scenarios and camera-only/multi-LiDAR scope remain pending. Report target remains 40–50 pages with combined figures.



## 20 October 2026 — quantitative nuScenes case analysis

- Hash-verified the preserved six-case input ZIP against its existing repository record; replayed all 12 model–case matching/FP/FN records exactly at score 0.30.
- Added exploratory 0.10/0.50 threshold checks across six frames (36 model–frame–threshold rows), plus class, shared-localisation and object-level ledgers for scene-0332/scene-0562.
- Seventeen of 18 discordant objects have no score-eligible same-class candidate within 2 m in the model that misses them; the remaining object has an eligible candidate assigned to neighbouring GT. All six highlighted misses recover at threshold 0.10. This diagnoses matching outcomes, not causes of learned score differences.
- Preserved one composite figure (PDF/PNG), CPU reproduction scripts, [methods/discussion draft](docs/report/CASE_ANALYSIS_DRAFT_2026-10-20.md), and [bounded evidence-gap status](docs/report/EVIDENCE_GAP_STATUS_2026-10-20.md).
- No inference, fresh SDK filtering or full-validation replay was run. Full replay, CLOCs root-cause investigation, wider literature review and sourced financial analysis remain open.



## 20 October 2026 — fresh full-validation ground-truth replay

- Rebuilt the Python 3.10.22 CPU evaluator environment and checked nuScenes-devkit 1.2.0 plus four evaluator source hashes against the original full evaluation.
- Rehashed both full prediction files; verified the original metadata backup and restored 19 metadata files. Replayed the official 6,019 samples / 150 scenes in separate model processes; exact official prediction/GT token equality and all filtered class counts passed.
- Both models reproduced all 32,451 compared numeric summary/detail values per model within absolute tolerance 1e-12, excluding evaluation time. Maximum absolute difference: 8.881784197001252e-16; zero comparison failures.
- Imported all 23 source ZIP members unchanged under [full-validation replay](results/nuscenes/bevfusion/full_validation/replay_20261020/README.md). Archive: 833,241 bytes; SHA-256 95eb6f99c1a495533bf62d1358fe7edc8527b7065ad032a7e646c2ad354b5a7a. Import rechecked member hashes, original reference bytes and numeric comparisons.
- This is same-evaluator reproduction, not independent cross-implementation verification. The repository import did not execute another evaluation. The checkpoint-linked upstream discrepancy and CLOCs root-cause analysis remain unresolved.



## 20 October 2026 — bounded CLOCs record audit and blend scope

- Audited 26 saved JSON records plus the current trainer against the repository snapshot. Equal recorded training inputs, fixed 96/32 split, source hashes and settings (except training seed/output folder), best-epoch/reload records and all nine paired baseline metrics pass.
- Seed-29 reaches development candidate baseline later (epoch 6 versus 4/3). Its final development candidate AP is 0.0209 points above seed-30 while Easy 3D AP40 is 7.2471 points lower. Cause remains inconclusive; no per-frame attribution or new AP40 evaluation performed.
- Identified a source reproduction gap: recorded Kaggle trainer SHA-256 23e798e0… differs from current generic repository trainer 7bb2cc59…. Exact executed source recovery remains required before a faithful rerun.
- Reviewed exploratory holdout-grid selection separately from the train-selected blend coefficient 1.875 (four-frame candidate AP 0.843266 vs 0.796576 baseline). Explicitly retained blend as unfinished exploratory pilot, without full-AP40 validation or a claimed seed-29 remedy.
- Saved [audit, two CSV tables, source hashes and discussion draft](results/kitti/clocs/audit_20261020/README.md) plus the standard-library reproduction script. Existing figures and original numerical records remain unchanged.



## 20 October 2026 — bounded nuScenes upstream-reference audit

- Streamed both published checkpoint URLs and computed complete SHA-256 values; sizes/hashes match all six saved inference protocols. No bulk weights added to Git.
- Compared both model dictionaries and test pipelines across three saved storage batches with the inherited pinned MMDetection3D configs: exact matches. Loader differences are data root, per-batch annotation path and explicit dataset version. Nine preceding sweeps and fusion image settings agree.
- Inspected linked upstream logs: LiDAR epoch-20 mAP/NDS ×100 64.85/69.56; fusion epoch-5 68.58/71.39. Recorded every logged validation row and contextualized the rounded model-table comparison. Both logs report 6,019 validation samples.
- Documented historical runtime/configuration differences and unresolved upstream SDK-version/annotation/checkpoint-log provenance. No cause established, no inference rerun and no attribution to hardware or randomness.
- Preserved [audit, field differences, source hashes, checkpoint download record and discussion](results/nuscenes/bevfusion/reference_audit_20261020/README.md), with a restricted-AST standard-library reproduction script. Original results remain unchanged.

## 28 October 2026 — completed case synthesis and arm clarification

- Consolidated A–D, original visibility/ten-class filtering, rainy success/regression and two preselected night frames into a supervisor summary and report discussion, with exact source links and a compact CSV.
- Rechecked arithmetic for all 48 A–D and 12 night threshold rows. Reaggregated all 45 preserved arm trial rows against nine summary groups and checked paired initial offsets. No inference, raw matching replay, official AP/NDS evaluation or physics rerun.
- Clarified the arm's identical light/heavy dimensions, common commanded path and mass/friction confound. Added one profile/placement/loading composite with explicit 0/5 heavy low-grip outcome; heavy feedback/high-fixed mean load comparison remains 16.08/38.63 N.
- Reconciled outdated pending-status statements in the case documentation. Ten-class counts and night analysis are complete; representative weather accuracy and positive annotated night/rain pedestrian evidence remain unestablished.
- Prepared a pinned-commit CPU utility that verifies downloaded public Git blobs, ZIP payloads and a destination copy. The user's Drive execution/receipt remains pending; no Drive completion is claimed.
- Outputs and reproduction: [supervisor follow-up](results/supervisor_followup/20261028/README.md). Report-wide chapter/literature integration and poster remain separate.

## 28 October 2026 — bounded additional night/rain pedestrian protocol

- Agreed one final annotation-positive pedestrian example from an explicitly night/rain-labelled scene before concentrating on dissertation writing. Existing two night cases remain intact.
- Froze scene/frame eligibility and alphabetical/midpoint selection independently of model outcomes, including official pedestrian categories, nonzero annotated point support, strict <40 m range and preserved part-10 keyframe availability.
- Prepared standard-library CPU recovery of metadata, seven sensor keyframes and verified saved full-validation predictions. Protocol and selection must be saved on Drive before predictions are read.
- Local synthetic checks passed for condition words, range/support exclusion, availability, selection tie-breaking and 7,000 Unicode records through a multi-chunk streaming TAR reader. No real frame has yet been selected or analysed.
- Actual Colab recovery and ZIP handoff remain pending. [Protocol and preparation](results/nuscenes/bevfusion/case_analysis/20261028/night_rain_pedestrian/README.md). No new GPU inference is planned.

## 28 October 2026 — additional annotation-positive night/rain pedestrian completed

- Verified uploaded ZIP SHA-256 1dcdef345a25febf7352cc393e06629e65c7935d4b4fd0fa3a47cd46b3a8ca97 and all 18 payload sizes/hashes. Inputs/prediction-subset preservation had already passed in Colab; original source archives were not freshly rehashed locally.
- Replayed the declared ranking from the preserved 120-frame candidate audit: scenes 1060/1065 have no eligible pedestrian frame; 1067 has 22. Frame 28e33ad25c0943b49b57f8f1346d5bd2 was selected before prediction access. GT-only projection selects CAM_BACK.
- Confirmed one eligible pedestrian at 30.0595 m, annotated visibility 80–100%, one current LiDAR point; raw-scan recomputation agrees. Both models miss at 0.10/0.30/0.50. Nearest scores 0.029148/0.008018; XY errors 0.212734/0.214623 m. At 0.30 both have 4 TP / 0 FP / 3 FN with identical matched GT identities.
- Added original-image paired figure (PNG/PDF), exact threshold tables, assignments/filter ledger, reproducible CPU script and bounded critical discussion. No inference or official AP/NDS replay. Retained both earlier night cases.
- Added a nine-unique-frame discussion map so all cases can be covered thematically without duplicate counts. Derived Drive copy remains separate from the verified input transfer; dissertation chapter integration is next.

## 29 October 2026 — first T4 fog smoke reviewed; fusion repeat diagnostic prepared

- Verified the uploaded first GPU attempt ZIP SHA-256 `5891509e3160a49f052bfed701a24ccc99e82bd6bd7150da9c14478438fbf189`, exact member set, CRC and all 31 payload sizes/hashes.
- Actual isolated setup/build and CUDA operation checks succeeded. LiDAR passed baseline and three fog conditions on the fixed pilot, including its original positional clean-repeat check.
- Fusion strictly loaded its checkpoint but stopped at baseline repeat agreement: 4/200 label positions differed. Original worker did not preserve the two fusion arrays before asserting, so root cause and permutation-only equivalence are not established. Full fog accuracy remains pending.
- Added bounded, environment-reusing diagnostic with input/cache/source/checkpoint verification, three baseline repeats per separate default/strict-deterministic subprocess, depth/branch feature hooks and complete prediction preservation. Original tolerance and smoke acceptance rules remain unchanged. Six local CPU comparison-helper checks pass; GPU diagnostic execution pending.

## 29 October 2026 — fusion repeat variation reproduced and deterministic execution validated on pilot

- Verified the uploaded diagnostic ZIP (176,119 bytes, SHA-256 `4ef30b9a808cabd729edc9ada7116a800ac52b7832a94def2d7ce6d4f51565b2`): all 23 payloads, CRC, sizes and hashes pass. Published complete derived diagnostic evidence and independent CPU recomputation script/results.
- Identical packed inputs produced differing projected-depth maps under default execution (5,410 / 5,124 changed elements); LiDAR-branch features remained identical within default mode. Default output equality was not recovered by permutation-aware comparison at the original tolerances.
- Strict PyTorch determinism gave exactly equal projected-depth and branch-feature hashes and exactly equal 200 boxes/scores/labels across three baseline forwards on T4. This is bounded same-session repeatability evidence; unique kernel attribution and all-condition validity remain unproven.
- Prepared a documented strict execution amendment and reusable paired smoke launcher. It verifies preserved caches/evidence/source/checkpoints/extensions, retains the original worker and numeric tolerances, and reruns both models at all four conditions without reinstalling. Fresh full-study baselines must use the same amended setting; old results remain retained. Strict paired smoke execution/review and full fog accuracy remain pending.

## 29 October 2026 — strict paired fog pilot accepted; full restoration prepared

- Verified accepted T4 smoke ZIP SHA-256 `a3127d6776fd175d5466a3d7272234efcdf3cd3df0aa60cff12d5ec739b3ba6d`, 206,111 bytes, exact member set/CRC and all 31 payload hashes. Published full derived evidence and local review.
- Both models passed baseline plus three fog conditions, original baseline positional repeats and strict execution checks. Rechecked all eight saved prediction arrays for finite/valid values and equal packed point hashes per level; fusion baseline exactly matches the earlier deterministic diagnostic.
- Accepted only the one-frame implementation gate. No full 600-frame inference or fog mAP/NDS exists yet; old default-execution results remain retained separately.
- Found immutable prior SHA-256 records for exactly the same 9,600 frozen sensor paths (4,700,293,684 bytes across 15 archives). Prepared an interruption-reusable selected-input restoration script with pinned provenance, fresh per-payload hashes, local write/reuse checks and Drive audit preservation. No fresh whole-archive hash claim, installation or inference. Six local helper/syntax checks passed; real full restoration pending.
