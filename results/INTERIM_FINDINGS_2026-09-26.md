# Interim findings from verified project artifacts — 26 September 2026

**Purpose:** Evidence-supported conclusions available for the technical project record. These are interim findings, not a final dissertation conclusion or a claim of completed learned fusion. All figures below come from the repository's numerical artifacts; data from distinct tasks have deliberately separate interpretations.

## Result 1 — LiDAR-only full KITTI baseline

The pretrained MMDetection3D PointPillars Car baseline achieved **90.3449 / 79.1299 / 76.1566% Easy / Moderate / Hard 3D AP40** at strict 3D IoU 0.70 over the frozen 3,769-frame KITTI validation split. BEV AP40 was **92.7663 / 88.3680 / 85.5054%**. This establishes a measured LiDAR-only reference for a later like-for-like CLOCs evaluation, not a camera–LiDAR fusion improvement. **Source:** `results/pointpillars_kitti_ap40_strict_summary.csv` and `results/pointpillars_evidence_manifest.json`.

## Result 2 — Camera detector trade-off on a small matched protocol

On the saved **10-frame camera holdout**, at 2D IoU 0.50, YOLOv8x had precision **0.581**, recall **0.800** and F1 **0.673**; Falcon with score-free deduplication had precision **0.528**, recall **0.844** and F1 **0.650**. On this operating point YOLOv8x had higher F1 and Falcon had higher recall. The experiment is a small, fixed-threshold comparison, not official KITTI camera AP or evidence that one model is always better. **Source:** `results/camera_detector_comparison.csv` and `results/VALIDATED_RESULTS.md`.

## Result 3 — Simpler projected-LiDAR depth method survived holdout

For **100 held-out frames**, among camera detections matched to KITTI Car annotations, median-ROI V1 had **1.953883 m MAE** (351 matched objects), nearest-cluster V2 **4.194522 m MAE** (350), and edge-aware V3 **1.961820 m MAE** (351). V3's slight development-set gain did not carry through in holdout MAE or RMSE; V1 was retained. These are *conditional matched-object depth errors*, and their unequal matched-object counts prevent a claim of paired statistical significance from this aggregate alone. **Source:** `results/depth_method_comparison.csv` and `results/VALIDATED_RESULTS.md`.

## Result 4 — Real CLOCs training labels exist, with severe candidate imbalance

Across **16 deliberately GT-stratified KITTI training frames**, the pipeline verified **1,714,176 aligned 3D candidates** and **2,191,437 2D/3D association rows**. Candidate labels comprise **1,345 positive**, **1,533 ignored** and **1,711,298 negative**: **1,272.34 negatives per positive**. There are **1,341/1,345 positives with a camera-associated row** and four positives without one. All 11 frames with retained Car GT have at least one positive; five frames have no retained Car GT. These counts support a class-imbalance-aware training design and validate the data pipeline; this curated 16-frame batch is **0.43% of the 3,712-frame training split**, cannot estimate whole-dataset prevalence, and yields no trained-fusion AP. One independent CPU geometry sample checked **258 candidates**, maximum absolute 3D IoU difference **2.97 × 10⁻⁸** against the independently implemented geometry; pinned CUDA/SECOND parity is still unverified after a native crash. **Sources:** `results/clocs_train_16_distribution_audit.json`, `results/clocs_train_16_training_example_audit.json`, and `results/clocs_000149_iou_scipy_audit.json`.

## Conclusions that remain unmeasured

- Whether learned CLOCs improves, matches or harms PointPillars on full held-out KITTI 3D AP40.
- Whether Azitt/PV-RCNN++, PointPainting-style fusion or BEVFusion improves detection on any common protocol.
- Whether KITTI findings transfer to nuScenes or withstand calibrated sensor corruption.
- Whether the final system meets a target runtime or is ready as an executable Glasgow demo.

The Mac inventory records a missing Azitt PV-RCNN++ checkpoint and absent nuScenes dataset files at checked paths. Source code or configurations alone cannot resolve those missing experiments. **Source:** `results/phase1_mac_dependency_inventory.json` and `docs/PHASE1_STATUS_AND_CRITICAL_PATH_2026-09-26.md`.

**Use in dissertation:** Quote the four measured interim results with their dataset, split, metric, denominator and limitation. Update this record only after a new run supplies its own numeric result and provenance. Never substitute camera F1, conditional depth MAE, pilot label counts, or LiDAR-only AP for measured fusion AP.
