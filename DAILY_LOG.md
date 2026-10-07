# Daily Project Log

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
