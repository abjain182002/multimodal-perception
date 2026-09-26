# Phase 1 status and critical path — 26 September 2026

**Project:** ENG5059P multimodal LiDAR–camera perception. **Evidence checked against:** the preserved 29 August locked-plan screenshot, `docs/PHASE1_DAILY_TARGETS_2026-09-25_TO_10-10.md`, `results/VALIDATED_RESULTS.md`, the 16-frame audits, and the Mac dependency inventory. The full transcript of the earlier “Use GitHub Repository-1” chat was not available; the screenshot and project plan preserve the scope described below.

**Target:** freeze the agreed technical work and its results by 8–10 October; start dissertation drafting after 10 October. Submission is 20 November, noon; poster event is 25 November. Dates are targets, not evidence of success.

## Research path

Establish separate camera-only, LiDAR-only and camera–LiDAR baselines on well-defined tasks. Train and evaluate learned CLOCs on KITTI using the **same 3D AP40 protocol** as the PointPillars baseline. Compare with the separately reproduced Azitt/PV-RCNN++ route and pursue PointPainting-style semantic fusion and BEVFusion; investigate nuScenes, degradation, ablations, runtime and a runnable software demo. Camera 2D results and matched-object depth error cannot be relabelled as 3D AP.

| Component | Status as of this record | Evidence / next gate |
|---|---|---|
| Frozen KITTI SECOND splits and GT | **Verified** | 3,712 train and 3,769 validation frames; 7,481-frame Car GT/range audit in `results/kitti_clocs_gt_audit.json`. Hold final validation out of fusion model selection. |
| Camera-only YOLOv8x, modern YOLO and Falcon | **Controlled pilots recorded** | Camera detector comparisons and score-free Falcon deduplication in `results/VALIDATED_RESULTS.md`; fixed operating points on small development/holdout samples, not full official camera AP. |
| Projected-LiDAR median-ROI depth | **Controlled development and holdout recorded** | V1/V2/V3 results and detector swaps in `results/VALIDATED_RESULTS.md`; matched-detection depth error, not 3D detection AP. |
| MMDetection3D PointPillars | **Full KITTI validation baseline measured** | Pretrained Car Moderate 3D AP40 **79.1299% at IoU 0.70**, 3,769 validation frames; `results/pointpillars_kitti_ap40_strict_summary.csv` and `results/pointpillars_evidence_manifest.json`. No learned fusion in this score. |
| CLOCs feature/target engineering | **Real 16-frame training pilot verified** | 1,714,176 aligned candidate boxes; 1,345 positive, 1,711,298 negative and 1,533 ignored candidate labels; `results/clocs_train_16_distribution_audit.json` and PNG/PDF. Independent 258-anchor CPU geometry check in `results/clocs_000149_iou_scipy_audit.json`. GT-stratified pilot is 16/3,712 train frames (0.43%), not a trained model or population estimate. Pinned CUDA/SECOND IoU parity did not complete because the native process crashed. |
| CLOCs train, selection and full KITTI evaluation | **Open: central scientific result** | Freeze a larger train-only corpus and separate train-only development IDs; fit and freeze the fusion checkpoint; generate full held-out KITTI predictions; compute comparable Car Easy/Moderate/Hard BEV and 3D AP40 plus failures/ablations. No CLOCs improvement or AP has yet been measured. |
| Azitt fusion baseline and OpenPCDet PV-RCNN++ | **Not reproduced** | `results/phase1_mac_dependency_inventory.json` verifies pinned source paths but reports the Azitt expected PV-RCNN++ checkpoint absent. A PointPillars result does not replace an Azitt/PV-RCNN++ run. Obtain the correct checkpoint and verify inference/evaluation on the frozen split. |
| PointPainting-style semantic fusion | **Planned, not measured** | Define inputs and training/evaluation on a common protocol; report a genuine run or an explicit blocker. |
| BEVFusion and nuScenes | **Planned, no dataset run established** | Pinned BEVFusion checkout/config present; Mac inventory finds no nuScenes dataset files. Secure eligible data/weights, validate taxonomy/calibration/splits, and run a clearly labelled within-dataset and/or transfer study. Never equate a mini smoke test with full generalisation. |
| MultiCorrupt, calibration/occlusion/range, ablations, runtime, demo | **Incomplete** | Run paired clean/corrupt experiments with denominators and provenance; time on named consistent hardware; save actual example outputs and an executable replay demo. Code/config presence alone is not a completed experiment. NVIDIA optimisation remains a stretch goal. |
| Figures and preservation | **Several real figures already saved; more required** | Existing PointPillars AP, distance, case-study, camera/depth comparisons and CLOCs pilot plots are inventoried in `docs/RESULT_FIGURE_INVENTORY.md`. Version small numerical outputs, scripts and reviewed PNG/PDF on GitHub at each milestone; store bulky inputs and weights separately with hashes. |

## Critical path, with exit evidence

1. **26–30 Sep: data and model gates.** Select/verify a larger train-only CLOCs batch, reuse the 16 validated examples, freeze an internal development partition *before* model selection, export real dense candidates, construct targets and verify candidate order/hashes. In parallel resolve Azitt weights and nuScenes/BEVFusion data availability; record immediate blockers.
2. **1–3 Oct: measured learned fusion.** Train CLOCs with valid class-imbalance handling and a recorded checkpoint, use only train-side development for choices, then evaluate once with the frozen protocol on the held-out KITTI validation split. Compare the exact eligible frames and AP settings with PointPillars; report negative results if observed.
3. **4–6 Oct: wider comparisons.** Run the separately pinned Azitt/PV-RCNN++ and planned semantic/BEV methods and the nuScenes experiment to the extent their verified resources allow; for each unrun part state the exact dependency rather than calling a substitute result complete.
4. **7–10 Oct: synthesis and freeze.** Complete controlled robustness/ablations, standardized runtime and software demo; recompute key tables, review original figures and qualitative case-selection rules, freeze source/checkpoint/metric manifests and write a completed/partial/blocked ledger. Dissertation drafting starts after 10 October.

## What progress claims mean

The 16-frame pilot proves the **CLOCs data pipeline produces real, auditable training labels**. It does not establish model learning, KITTI fused AP, cross-dataset transfer or completion of the whole project. A past “80% of Phase 1” estimate cannot be substantiated from the experiment evidence and must not be used as a scientific progress metric. As of 26 September, the principal trained-fusion comparison and several locked-scope methods remain open. GPU limits, missing Azitt weights and missing nuScenes data place the 8–10 October technical target at material risk; surface blockers to the supervisor promptly if unresolved.

**Assessment context:** The user-provided University of Glasgow marking schema assigns 25% to technical work, 60% to dissertation and 15% to poster. A1 is an ambition, not a promised grade. Scientific claims must remain linked to actual experiments and uncertainty.
