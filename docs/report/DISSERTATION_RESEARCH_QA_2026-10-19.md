# Dissertation research Q&A — corrected against completed project evidence

**Prepared: 19 October 2026.** Evidence reviewed at repository commit `3228ffa72eac7c72bd246dc7d3f039f0e77941f4`. This document answers the supplied 28 questions and separates measured results, literature-based interpretation and proposed work. It is a working dissertation guide, not evidence of new experiments or a completed literature review.

**Metric convention:** mAP and NDS values below are multiplied by 100. Differences are percentage-point changes unless explicitly described as ratios. Rounded displayed values can differ slightly from differences computed at full precision.

## Project position and evidence already available

The project investigates camera–LiDAR perception through KITTI baseline/fusion/depth studies and nuScenes pretrained BEVFusion evaluation, controlled input challenges and runtime measurement. Its contribution is an auditable comparative experimental study, including limitations and negative results. It has not established a new state-of-the-art architecture.

| Evidence stream | Completed evidence | Boundary |
|---|---|---|
| KITTI detection | PointPillars, PV-RCNN and CLOCs-related evaluations and seed diagnostics | Training histories and runtime protocols are not all matched |
| Camera and depth work | YOLOv8-based camera work and ROI-depth development/holdout comparisons | Not a camera-only nuScenes 3D benchmark; not an exhaustive YOLO comparison |
| Full nuScenes validation | Both pretrained pipelines on 6,019 samples, 150 scenes | Official validation, not hidden test; no retraining |
| Camera availability | Four conditions on frozen 600 samples across 150 scenes | Masked views retain six model input slots; not a physical two-camera deployment |
| Image darkening | Clean, 0.5 and 0.25 image-value multipliers | Synthetic darkening, not real-night evaluation |
| LiDAR point reduction | Both models at 100%, 50% and 25% retention | Uniform thinning, not different physical LiDAR hardware |
| Blur, noise, calibration | Seven conditions, predictions and metrics preserved | Two severities per challenge; no threshold or significance established |
| Runtime | 150 frames, three sessions per model, single T4 | Specific timing boundary; not embedded deployment or energy measurement |
| Qualitative evidence | Six annotated case figures and broader candidate analysis | Illustrative cases, not causal proof or population frequencies |
| Reporting assets | Architecture diagrams, compact figures, scripts and GitHub evidence | Final dissertation assembly and citation audit remain |

Evidence entry points: [validated results](../../results/VALIDATED_RESULTS.md), [page/evidence plan](EVIDENCE_AND_PAGE_PLAN.md), [architecture](../architecture/README.md), [camera challenges](../../results/nuscenes/bevfusion/camera_challenges/20261016/FINDINGS.md), [runtime](../../results/nuscenes/bevfusion/runtime/20261015/FINDINGS.md), [cost framework](../../results/cost_performance/README.md).

**Essential corrections to the supplied draft:** BEV fusion is a form of feature fusion; core reading is not the same distinction as primary/secondary sources; three yaw settings cannot locate a failure threshold; camera-only corruption inference need not involve retraining; multi-LiDAR processing need not require a new network branch; masking does not measure hardware savings; published methods cannot be placed on a shared Pareto frontier using incompatible protocols; and an A1 grade cannot be guaranteed by a paper count or checklist.

## Task 1 — Literature and state-of-the-art review

### Q1.1 — How should the literature review support the experimental narrative?

Organise it around the research questions: when fusion helps, when it fails, what computation it costs and what sensor configurations have actually been tested. Use a compact taxonomy with three descriptors:

- **Fusion stage and representation:** candidate-level fusion; feature fusion in point/image space; feature fusion in BEV. BEV is a representation within feature fusion, not a mutually exclusive fusion level.
- **Inputs at inference:** camera-only, LiDAR-only or combined, with physical sensor counts and temporal inputs stated separately.
- **Robustness role:** nominal detector, corruption benchmark, robustness mitigation or calibration-estimation method.

Connect candidate fusion to the KITTI CLOCs work, BEV fusion to the nuScenes study, and robustness literature to the controlled challenges. Papers can occupy multiple categories. End each subsection with a specific evidence gap and the experiment that addresses it; do not force unrelated papers into a single grid.

### Q1.2 — Which papers need substantive engagement?

Use **core reading versus supporting reading**. Original research papers remain primary sources even when used only for context.

Core reading should cover the methods actually evaluated—PointPillars, PV-RCNN, CLOCs, the implemented BEVFusion lineage—and **both KITTI and nuScenes**. Read the two BEVFusion papers distinctly. Add GraphBEV, MetaBEV, Robo3D, RoboBEV and Dong et al.'s *Benchmarking Robustness of 3D Object Detection to Common Corruptions* for the robustness discussion [R1–R6, R9].

BEVFormer and BEVDepth become core to the camera-only feasibility decision [R7–R8]. CenterPoint, Fast-CLOCs, DeepFusion and TransFusion support architectural comparison; describe them as literature comparators unless separately evaluated. LCCNet and Waymo support calibration and multi-LiDAR scope [R10–R12].

Around 15–20 carefully used research papers is a practical starting set, not a marking requirement. Add newer papers only when they change a claim or decision.

### Q1.3 — How can claim attribution be made defensible?

Create a claim register containing: claim ID, exact proposed sentence, paper version, section/table/figure, dataset/split, training and inference conditions, supported interpretation and limitation. Keep project evidence identifiers alongside literature references.

Examples of corrections:

- Liu et al.'s ICRA 2023 BEVFusion and Liang et al.'s NeurIPS 2022 BEVFusion are different works [R1–R2].
- GraphBEV's Figure 1 compares 69.1 versus 60.8 mAP under its noisy setting: **8.3 percentage points**, not automatically an 8.3% relative improvement. Its LocalAlign and GlobalAlign address different alignment stages; its experiment is not our yaw protocol [R3].
- RoboBEV discusses pre-training, depth-free transformations and temporal information as robustness factors in its evaluated models; this does not establish universal causal effects in our checkpoint [R5].

This Q&A verifies selected disputed claims, not every page of every suggested paper.

### Q1.4 — What state-of-the-art claim can this dissertation make?

Use: **“This dissertation evaluates established pretrained perception pipelines and characterises their accuracy, computational requirements and sensitivity to controlled input changes.”**

For the full 6,019-sample validation set, LiDAR-only achieved **64.4301 mAP / 69.2353 NDS**, and fusion achieved **68.3099 / 71.1352**, increases of **3.8798 and 1.8998 points**. These support a comparison of the evaluated checkpoints, not a claim to beat current methods.

Do not call a 2022–2023 method the October 2026 state of the art without a dated, current literature and benchmark review. Published scores require separate provenance columns for split, backbone, training data, temporal input, augmentation and ensemble use. Our literature comparison and our experimentally matched comparison should remain visibly distinct.

## Task 2 — Failure-handling synthesis

### Q2.1 — How should observed failure modes be organised?

Use intervention categories and a separate mechanism/interpretation column:

| Intervention category | Project evidence | What it does not establish |
|---|---|---|
| Image information degradation | Darkening, Gaussian blur, additive noise | Real weather/night or physical camera-noise equivalence |
| LiDAR density reduction | Matched 50%/25% point retention | Weather robustness or a cheaper LiDAR's performance |
| Image availability reduction | Front/back-only information and front-camera masking | Missing physical hardware, camera-only inference or total LiDAR failure |
| Geometric inconsistency | Common camera-rig yaw offsets | Every calibration fault or an exact failure threshold |

Treat over-reliance on a modality as a **hypothesis about mechanisms**, not an independently demonstrated fifth experiment. A failure can involve several mechanisms. Do not force every observation into exactly one causal category. Use the six annotated cases to illustrate measured patterns and mark unconfirmed explanations explicitly.

### Q2.2 — How should the yaw result be presented as a contribution?

Describe the actual intervention: for column-vector transforms, `C2L' = Rz(theta) C2L`, `L2C' = inverse(C2L')`, and `L2IMG' = K4 L2C'`. The common rotation is about the **LiDAR-frame +Z axis and origin**, so camera translation vectors also rotate. It is not an independent rotation about each camera's optical centre. LiDAR points, images, intrinsics and ground-truth/output coordinates remain unchanged.

| Common yaw offset | mAP | NDS |
|---|---:|---:|
| 0° | 68.1018 | 71.2767 |
| +1° | 67.4287 | 70.8153 |
| +3° | 64.6339 | 69.3172 |

This demonstrates sensitivity at the tested offsets on 600 frames. It does **not** locate a threshold, prove a cliff or test negative rotations. Connecting markers is only a visual guide.

GraphBEV supplies an alignment-related interpretation, but the perturbation may affect depth projection and BEV alignment; it is not proven to isolate its “global” mechanism [R3]. Alignment correction and LCCNet-style calibration estimation are possible future mitigations [R10], not demonstrated remedies here.

### Q2.3 — What can be said about sensor failure?

The NeurIPS BEVFusion paper explicitly motivates a camera stream independent of LiDAR input [R2]. MetaBEV investigates adaptation to available modalities [R4]. These motivate robustness-aware designs; their reported numbers require their exact failure settings and baselines.

Our evidence is narrower. The evaluated MMDetection3D fusion configuration uses LiDAR-derived depth in its camera pathway, and camera masking preserves six view slots. This warrants discussing dependency, but **total LiDAR absence was not tested**. Do not assert a measured collapse, automatic fallback or successful camera-only operation.

Architecture can enable operation with missing inputs; useful accuracy also depends on training and implementation. Compare the design choices, then clearly state the missing experiment. Do not generalise every property of our exact configuration to all implementations called BEVFusion.

### Q2.4 — How does point reduction connect to corruption research?

Frame it as a controlled density-sensitivity experiment motivated by broader corruption benchmarks [R6, R9]. The same selected point subsets were supplied to both models. At 25% retention, LiDAR-only obtained **51.0381 mAP / 61.1077 NDS**, while fusion obtained **60.2341 / 65.4790**, versus their respective full-density baselines of **64.3076 / 69.4430** and **68.1302 / 71.2891**.

Fusion lost less accuracy under this protocol. However, thinning can alter both LiDAR features and LiDAR-derived camera depth. It does not isolate one branch.

Uniform random thinning does not reproduce spatially structured rain, snow, dust, beam patterns or changed hardware. It characterises one intervention; it does not mathematically bound the whole density-corruption space. Preserve the selection seed and matched-index evidence.

## Task 3 — Cost versus performance

### Q3.1 — How can a credible cost model be built?

Separate **measured computational cost** from **hypothetical procurement scenarios**. The existing cost register is a framework, not a completed bill of materials.

For each physical scenario, record product, specification, region, currency, date, quantity, taxes, included optics/interfaces and source. Model sensor acquisition, compute, integration, synchronisation, storage and maintenance separately. An integration allowance is an explicitly varied assumption, not an observed 15–25% industry constant.

Quote-only products remain “price unavailable” until sourced. Historical hobby-camera prices cannot stand in for complete automotive camera systems. Hardware with different optics or scan patterns cannot inherit nuScenes accuracy.

A one-LiDAR/two-camera accounting scenario is legitimate; assigning the masked six-view model's accuracy and runtime to that physical system is not.

### Q3.2 — Which performance metrics and protocol details accompany cost?

Use mAP/NDS, latency and peak memory where measured. Parameter count or FLOPs are optional additions only with a reproducible counting convention and coverage of custom operations.

Our measured runtime is **single Tesla T4, batch size 1, FP32, autocast off**:

| Pipeline | Median of three session medians, ms | Maximum peak allocated PyTorch memory, MiB |
|---|---:|---:|
| LiDAR-only | 287.8116 | 609.8184 |
| Camera–LiDAR | 544.9786 | 2231.7979 |

The primary latency covers `test_step`; CPU loading/collation is outside it. Each model used 150 frames and three isolated sessions. The 600-frame accuracy context is a **different sample set/protocol**, not accuracy recomputed during timing. The fusion latency ratio is approximately **1.8935×**.

The KITTI PV-RCNN timing boundary differs; do not place it in the same ranking without a matched protocol. Allocated memory is not device-wide memory or energy.

### Q3.3 — How should the trade-off be analysed?

Start with the measured nuScenes accuracy–latency trade-off, explicitly identifying the separate 600-frame accuracy context. Fusion improves accuracy but increases latency and memory. With only these two systems, report the observed trade-off; avoid implying a comprehensive frontier over modern methods.

A Pareto frontier is meaningful only within a specified comparable set and objectives. Do not pre-assign PointPillars, PV-RCNN and fusion to supposed positions. KITTI AP40 and nuScenes mAP/NDS are not a common accuracy axis; published runtimes on other hardware are not interchangeable with our T4 measurements.

Keep monetary scenario plots separate until prices and mappings are defensible. Include sensitivity ranges for assumptions. Two nondominated points do not demonstrate a market-optimal sensor package.

### Q3.4 — Does this require a GPU?

The remaining cost analysis, tables and plots can run on CPU using preserved records. No GPU rerun is needed merely to redraw figures or calculate ratios.

Missing runtime or memory measurements must be marked unavailable. Exclude unsupported methods from quantitative trade-off plots, or give them a separate literature-only table. A new matched benchmark could be justified if it answers a specific unresolved question, but it should not be triggered automatically.

Cloud T4 latency does not measure energy use, automotive deployment cost or guaranteed real-time performance. Those require additional measurements or separately labelled scenario assumptions.

## Task 4 — Camera-only 3D feasibility

### Q4.1 — Is a fair comparison practical?

A bounded **pretrained-system comparison** may be practical, conditional on checkpoint access, compatible software, memory, temporal inputs and data availability. Use identical target tokens, ground truth, evaluator, class definitions and intervention definitions.

Equal training budgets are necessary for a tightly controlled architecture/training ablation, but cannot usually be reconstructed for unrelated public checkpoints. Record such differences instead of claiming a pure modality effect. Keep each model's documented input requirements and disclose resolution/history differences.

**Robustness evaluation does not inherently require retraining.** A frozen camera-only model can be evaluated under corruption, just as our fusion model was. Retraining concerns mitigation or configuration optimisation.

This remains untested. The present YOLOv8 and ROI-depth work does not substitute for camera-only nuScenes 3D detection.

### Q4.2 — What must be checked before running it?

Read BEVFormer's spatial/temporal design [R7] and BEVDepth's depth-estimation and multi-frame design [R8]. Then write a feasibility receipt covering:

1. Official checkpoint/configuration, training split and reported validation metric.
2. Licence and dependency compatibility.
3. Required cameras, image resolution and historical frames.
4. One-sample GPU smoke check, memory and an initial runtime estimate.
5. Frozen evaluation tokens, clean-baseline validation and export/backup plan.

A downloadable checkpoint does not guarantee cheap inference. Historical images may exceed the current 600-frame transfer package. Prefer one defensible baseline rather than training from scratch or testing many models. If infeasible, document the actual blocker and present literature results separately. Do not claim resource impossibility before checking.

### Q4.3 — What is the main caveat if it is evaluated?

The comparison is between pretrained systems, not an isolated causal test of depth sensing. Depth uncertainty is an important interpretation supported by BEVDepth's motivation [R8], but performance also depends on architecture, training, history, image quality and calibration.

Specify **camera-only at inference** separately from any LiDAR-derived supervision used during training. A model can use depth supervision during training without using LiDAR at inference.

Do not declare depth the binding constraint solely from a lower aggregate score. Analyse classes, ranges or localisation errors where supported. LiDAR fusion supplies geometric information but does not eliminate sparse-depth or calibration limitations—the current yaw results make that distinction relevant.

### Q4.4 — How should a camera-only gap be interpreted?

Do not assume the result in advance. If camera-only underperforms, report the aggregate gap and then examine support-counted class or distance groups using the same evaluator conventions. If it improves under a specific challenge, distinguish absolute accuracy from loss relative to its own clean baseline.

Avoid asserting that a persistent fusion advantage proves a non-depth mechanism. Aggregate metrics do not identify causes. Likewise, a gap that narrows does not prove the camera model has become intrinsically more robust.

A negative or inconclusive result can still answer the supervisor's question if the protocol, uncertainty and limitations are explicit. These analyses remain proposed until predictions and evaluation outputs exist.

## Task 5 — Physical multi-LiDAR feasibility

### Q5.1 — What counts as separate LiDAR inputs?

Two or more physical sensors, with sensor identity, extrinsics and timing preserved. Multiple sweeps from one sensor, point thinning and splitting a point cloud do not qualify.

The Waymo Perception documentation describes five LiDARs: one top mid-range sensor and four short-range sensors [R11]. It is a relevant candidate among the datasets discussed. Select an exact release and verify its sensor fields; do not reuse historical scene counts or diversity slogans as current dataset facts.

The project's nuScenes inputs use one physical LiDAR, and the KITTI study does not supply a physical multi-LiDAR comparison. Sensor count alone is insufficient: placement, type, coverage and timing affect the interpretation of any incremental benefit.

### Q5.2 — Do compatible models exist, or is new architecture mandatory?

The draft's “models do not exist” conclusion is too strong. OpenPCDet's Waymo processing transforms and concatenates points from multiple LiDARs into a common vehicle frame and records per-sensor point counts [R12]. This demonstrates a standard merge route; a separate encoder branch is not inherently required.

A particular model/checkpoint's availability, licence and exact preprocessing still require checking. No suitable checkpoint has been validated for this project's proposed comparison.

A fixed-model sensor-subset ablation may be possible without retraining; it measures sensitivity to removing sensors from the training configuration. Training an optimised single- versus multi-sensor model answers a different question. Do not describe either as completed or necessarily cheap.

### Q5.3 — How can the professor's question be answered without running it?

Provide a feasibility design, not an invented result:

- **Concept:** transform synchronised sensor returns to a common frame, preserve sensor IDs and compare declared sensor subsets. Candidate-level fusion is another design option, but CLOCs' camera/3D candidate features do not automatically generalise to arbitrary LiDAR sets.
- **Evidence:** Waymo offers relevant sensor data; our existing experiments do not quantify the gain from extra physical LiDARs.
- **Proposed measurement:** compare top-only, a justified added-sensor subset and all available sensors on the same held-out frames, using Waymo AP/APH, latency and memory. Account for placement and sensor-type differences.

Cost scenarios can accompany this design, but cannot claim an empirical accuracy gain. Mark it “feasibility unresolved” until a documented decision is made with the supervisor.

### Q5.4 — What would justify GPU work?

Require verified data access and sensor identities; a compatible checkpoint and known preprocessing; an auditable sensor-subset implementation; and a small runtime/storage estimate that fits the remaining schedule. Freeze the hypothesis, subsets, metrics and stopping criteria before full inference.

Use **Waymo metrics**, not nuScenes NDS, for a Waymo experiment. A useful question is whether additional sensors improve AP/APH under a fixed model and what latency changes accompany that intervention.

A training budget is required only if the chosen question requires training. Do not gate a valid inference-only study on unnecessary retraining. If a prerequisite fails, record the concrete reason and a future-work design, rather than asserting that multi-LiDAR research is impossible.

## Task 6 — Tables, figures and report writing

### Q6.1 — What is the minimum compact presentation?

Use the existing assets selectively. For a 40–50-page target, a practical core is:

| Asset | Content |
|---|---|
| Literature table | Method, input/representation, robustness role, relevance and limitation |
| Main results table | Separate KITTI and nuScenes panels with their own metrics/splits |
| Robustness table | Conditions, clean baseline, absolute scores and within-study changes |
| Compute/cost table | Measured T4 results separated from sourced monetary assumptions |
| Architecture figure | Implemented pipelines, reused components and intervention points |
| Composite robustness figure | Panels for availability, darkening, point retention, blur/noise and yaw |
| Compute trade-off figure | Accuracy context versus measured latency/memory, sample sizes explicit |
| Compact case figure | A small selection of readable annotated successes and failures |

These are design targets, not a universal minimum. Use readable full-width panels rather than squeezing every case into one image. Put complete metrics and additional figures in the repository. Captions need protocol and supported interpretation; irrelevant hardware detail need not appear in every conceptual caption.

### Q6.2 — How should the discussion synthesise results?

Use claim–evidence–limitation–implication, but ensure the claim follows the actual data. For example:

“Fusion is sensitive to the tested camera-rig calibration perturbation. Increasing the offset from 0° to +3° reduced mAP from 68.1018 to 64.6339 and NDS from 71.2767 to 69.3172 on the frozen 600-frame set. Only two positive offsets were evaluated, and no calibration-recovery method was tested. The result motivates evaluating alignment or calibration correction under a matched protocol.”

Do not write “calibration is the dominant failure mode”: noise_std25 produced a larger loss in this study, and the severities are not physically matched across interventions.

Also discuss the mixed CLOCs seed outcomes and the ROI-depth holdout result. They show that adding fusion complexity or improving a development metric does not guarantee held-out improvement.

### Q6.3 — What is the CPU figure workflow?

Generate tables and figures from the same preserved metric records using reproducible scripts. Several such plotting scripts already exist; extend or assemble them instead of independently transcribing numbers.

Keep original predictions, metrics and manifests immutable. If an error is found, identify whether it is in evaluation or derived analysis, create a corrected version with provenance, and regenerate dependent outputs. **Do not hand-edit a raw metric CSV merely to repair a figure.**

Export vector PDF for suitable plots and PNG for previews, then inspect at final report size for text, legends and colour/grayscale readability. Record script version and input hashes. A combined figure saves space only if its panels remain legible.

### Q6.4 — In what order should writing proceed?

Write methods/setup and the evidence inventory first. Next stabilise tables, captions and findings; then write the discussion and refine the introduction/literature narrative around the supported contributions. Maintain a literature outline throughout rather than delaying all reading until the end.

Draft the abstract and conclusion early and revise them after numerical reconciliation—do not leave their first draft to submission day.

Protect a 1–2-hour writing block alongside technical work. The 40–50-page goal and 16–20 November window are user-reported; verify the institution's actual rules, reference style and submission date. The existing page plan is provisional. Compact presentation should remove repetition, not remove inconvenient results or methodological limitations.

## Task 7 — Final evidence and reference checks

### Q7.1 — What numerical checks are needed?

Create a claim-to-source map from report sentence/table cell to metric key, experiment protocol and source file/hash. Recompute rounding, changes in percentage points and ratios from full precision.

Check split, sample count, filtered GT, class list, metric definition, checkpoint, seeds, precision and timing boundary. Preserve separate labels for 6,019-frame validation, 600-frame robustness and 150-frame timing. Check KITTI IoU threshold and AP40 difficulty levels separately from nuScenes metrics.

Numbers need not appear in both a table and figure: every numerical claim needs a traceable source. Do not duplicate them solely for compliance.

Distinguish archive integrity verification, arithmetic verification of stored metrics and a fresh prediction-versus-GT evaluator replay. Passing one does not imply the others.

### Q7.2 — What citation checks are required?

Verify titles, authors, venue, year and the exact cited version. Maintain distinct bibliography entries for the ICRA and NeurIPS BEVFusions [R1–R2]. Check GraphBEV's ECCV 2024 record, MetaBEV's ICCV 2023 record, BEVDepth's AAAI 2023 record, LCCNet's **CVPR Workshops** 2021 record and RoboBEV's TPAMI 2025 version [R3–R5, R8, R10].

For each substantive attributed claim, check the actual method or results section, including baseline and protocol. Do not inherit an unverified number from another paper's related work.

Follow the required institutional citation style: **IEEE numbered references are normally ordered by first citation, not alphabetically**. Resolve links/DOIs and remove uncited entries unless a separate bibliography is permitted. A full reference audit remains pending.

### Q7.3 — What does preservation mean here?

Preserve enough evidence to trace and, where permitted, reproduce the claims: scripts, configs, environment records, metric files, protocols, hashes, evaluator details and instructions linking scripts to report outputs.

The project already preserves compact evidence in GitHub and raw prediction exports/backups separately on Drive. The seven-condition evidence archive has SHA-256 `2ed35d5e05024a5997d2485aa7860c45275208efcda0340c9f6dde02af8386be`; its raw inference export has SHA-256 `7bd97d3032515922d12e770b663e226bacc8e338e096182b9f600f063d32392a`.

Verify supervisor access to the repository; a private repository needs the correct access arrangement. Do not make licensed sensor data public merely for reproducibility. Document acquisition/access requirements and exclusions. A tagged release or institutional/DOI archive is useful if permitted, but not a substitute for working instructions. Clean-clone figure regeneration remains a final acceptance check.

### Q7.4 — What is the final checklist?

1. Reconcile report claims against saved results and protocols.
2. Complete paper-claim, bibliography and link checks.
3. Regenerate selected tables/figures from a clean checkout.
4. Inspect captions and layouts at actual report size.
5. Verify every claimed contribution and distinguish reused architecture from project work.
6. State missing experiments, uncertainty and generalisation limits without inventing infeasibility.
7. Confirm supervisor repository access and long-term evidence availability.
8. Check institutional format, page/word rules, ethics/data requirements and submission date.
9. Finalise abstract/conclusion against the verified results.
10. Leave a proofreading and submission buffer.

Completion of experiments is not completion of the dissertation or every supervisor request. Do not mark the literature synthesis, monetary analysis, camera-only/multi-LiDAR decisions or final report QA complete until their deliverables exist.

## Additions worth making now

### Priority 1 — Strengthen existing evidence without new inference

- **Claim/evidence register:** connect each major report conclusion to a result file and a primary-paper locator.
- **Within-study baseline discipline:** the camera-challenge clean result is 68.1018/71.2767, while the darkening and point-reduction clean values differ slightly. Use each study's own clean result; investigate documented execution differences before pooling. Do not silently substitute one baseline.
- **Uncertainty analysis:** consider paired scene-level bootstrap intervals for selected differences. Resample scenes, retain all their frames and recompute the metric correctly; naive per-frame AP averaging is invalid. Repeated scenes require multiplicity-aware evaluation/token handling. This is proposed CPU work, not existing significance evidence.
- **Support-aware interpretation:** include class counts and avoid generalising from sparse classes or selected cases. Range analysis needs a declared evaluation/matching definition.
- **Negative findings:** retain CLOCs seed variability and the ROI-depth holdout comparison. V1 holdout MAE/RMSE were 1.953883/3.924869 versus V3's 1.961820/3.947876; the development improvement did not translate to that holdout.
- **One contribution statement:** “An auditable comparison of established camera–LiDAR pipelines, with controlled robustness interventions, measured computational trade-offs and explicit limits on sensor-configuration conclusions.”

### Priority 2 — Close the supervisor's remaining questions

Complete the critical literature/failure-handling synthesis and sourced cost scenarios. Produce explicit camera-only and physical multi-LiDAR feasibility decisions. Neither is already answered experimentally. If a bounded camera-only checkpoint evaluation is practical, it is more directly compatible with the existing nuScenes evidence than an unplanned broad model survey.

An exhaustive comparison of “all YOLO models” is not necessary to justify YOLOv8. Explain its role, version, checkpoint, operating point, integration and limitations. A small matched comparator study is optional and must not be presented as completed.

### Priority 3 — Add experiments only for a specific claim

Additional yaw angles/signs would be needed before investigating a threshold. Multiple noise seeds would support perturbation variability analysis. Robustness training or an alignment module would test mitigation, a new question beyond the present sensitivity study. These are options, not mandatory requirements for a strong report.

No new GPU work, statistics or financial quotes were produced in preparing this document.

## Primary-source anchors and claim locators

These sources support corrections and provide reading anchors, not a claim that all suggested literature has been fully reviewed. Project measurements come from the linked repository evidence, not these papers.

| ID | Source | Relevant locator/use |
|---|---|---|
| R1 | [Liu et al., BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation, ICRA 2023](https://hanlab.mit.edu/projects/bevfusion) | Official project and paper; distinguish implementation/configuration from general paper claims |
| R2 | [Liang et al., BEVFusion: A Simple and Robust LiDAR-Camera Fusion Framework, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/43d2b7fbee8431f7cef0d0afed51c691-Abstract-Conference.html) | Abstract/method motivation: LiDAR-independent camera stream |
| R3 | [Song et al., GraphBEV, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03796.pdf) | Abstract, introduction and Figure 1; local/global alignment and noisy-setting comparison |
| R4 | [Ge et al., MetaBEV, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Ge_MetaBEV_Solving_Sensor_Failures_for_3D_Detection_and_Map_Segmentation_ICCV_2023_paper.html) | Method and sensor-failure evaluation; [official implementation](https://github.com/ChongjianGE/MetaBEV) |
| R5 | [Xie et al., Benchmarking and Improving Bird's Eye View Perception Robustness in Autonomous Driving, TPAMI 2025](https://arxiv.org/abs/2405.17426) | Abstract and version record; [journal DOI](https://doi.org/10.1109/TPAMI.2025.3535960) |
| R6 | [Kong et al., Robo3D, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Kong_Robo3D_Towards_Robust_and_Reliable_3D_Perception_against_Corruptions_ICCV_2023_paper.html) | Corruption-suite definitions; do not equate our thinning with the complete suite |
| R7 | [Li et al., BEVFormer, ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/694_ECCV_2022_paper.php) | Spatial/temporal design and checkpoint input requirements |
| R8 | [Li et al., BEVDepth, AAAI 2023](https://ojs.aaai.org/index.php/AAAI/article/view/25233) | Abstract and depth modules; 60.9 NDS headline is on the test set, not our validation subset |
| R9 | [Dong et al., Benchmarking Robustness of 3D Object Detection to Common Corruptions, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Dong_Benchmarking_Robustness_of_3D_Object_Detection_to_Common_Corruptions_CVPR_2023_paper.html) | Directly relevant camera/LiDAR corruption benchmark; useful addition to supplied reading list |
| R10 | [Lv et al., LCCNet, CVPR Workshops 2021](https://openaccess.thecvf.com/content/CVPR2021W/WAD/html/Lv_LCCNet_LiDAR_and_Camera_Self-Calibration_Using_Cost_Volume_Network_CVPRW_2021_paper.html) | Calibration estimation; not evidence of mitigation in our experiments |
| R11 | [Waymo Perception dataset documentation](https://waymo.com/intl/fil/open/data/perception/) | Sensor suite; five physical LiDARs; verify selected release before experimentation |
| R12 | [OpenPCDet Waymo preprocessing](https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/datasets/waymo/waymo_utils.py) | `convert_range_image_to_point_cloud` and `save_lidar_points`: common-frame concatenation and per-LiDAR counts |

Primary-source pages/code checked 19 October 2026. Moving code references must be pinned to a commit before reuse in an experiment. The final dissertation bibliography must additionally include the original KITTI, nuScenes and evaluated detector papers, using verified bibliographic records.
