# Report evidence and page plan — 15 October 2026

## Agreed presentation constraint

The user targets **40–50 pages**, including substantial explanatory content, results, tables and figures. This is a planning preference, not a verified university page-count rule. Confirm whether title pages, references and appendices count before fixing layout. Submission is user-reported as 16–20 November; official deadline confirmation remains necessary.

Combine related plots into labelled panels at readable final page size. Use compact tables for exact values; captions must explain inputs, metrics, samples, units and limitations. Preserve complete evidence and detailed class plots in GitHub, with references from the report. Do not discard negative results, omit limitations, mix KITTI/nuScenes scores or shrink text to fit.

## Provisional 46-page allocation

| Section | Pages |
|---|---:|
| Front matter | 3 |
| Introduction, objectives and contributions | 3 |
| Literature review | 5 |
| Methodology, architecture and implementation | 10 |
| Results, robustness and case studies | 12 |
| Discussion, limitations and cost analysis | 5 |
| Conclusions and future work | 2 |
| References | 4 |
| Selected appendix material | 2 |
| Total | 46 |

Figures/tables occupy these allocations; they are not additional pages. This is flexible once the formal template and counting rules are verified. The earlier 1,200-word literature-review request should be reconciled with the final chapter plan rather than silently expanded.

## Evidence coverage

| Requirement | Position at this checkpoint | Report work remaining |
|---|---|---|
| KITTI LiDAR and fusion comparisons | Full validation and seed diagnostics preserved | Explain task/metric differences and small-data limits |
| Full nuScenes | 6,019 frames / 150 scenes preserved | Explain merged evaluation, pretrained comparison and limits |
| Architecture | Study overview and pinned BEVFusion diagrams preserved | Distinguish reused architecture from own integration/analysis |
| Failure cases | Six annotated cases preserved | Select representative cases, relate cautiously to literature |
| Camera availability | 600-frame study complete | Missing-view robustness, not physical two-camera cost |
| Image darkening | Three-condition study complete | Synthetic brightness, not real-night conditions |
| LiDAR point reduction | Six conditions complete | Matched inputs, pipeline-specific own baselines |
| Blur and noise | Planned, not completed | One defined corruption per family, two levels, frozen protocol |
| Calibration errors | Planned, not completed | Define a consistent transform intervention and two magnitudes |
| Runtime/memory | 150-frame paired three-repeat study complete | Timing boundary, hardware, accuracy context and caveats |
| Monetary cost | Framework/source register only | Scenario, comparable hardware quotes, assumptions and sensitivity |
| Literature/SOTA | Draft available separately, not finalized | Verify primary papers, recent coverage, critical comparison and failure-handling evidence |
| Camera-only 3D / physical multi-LiDAR | Unresolved scope | Establish feasibility or agree documented limitation with supervisor |
| Discussion/conclusions | Evidence available, chapter not complete | Answer research questions and compare against literature |

## Compact figure strategy

- Architecture: combine study scope and relevant model detail only if both remain readable; otherwise use two purposeful diagrams.
- Benchmarks: one KITTI composite and one full-nuScenes composite; preserve metric distinctions.
- Robustness: group camera challenges and geometric challenges into a small number of composite figures after all planned tests finish. Point reduction already has one two-panel main figure.
- Case studies: a few clearly annotated representative successes and failures, with exact matching rules and no unverified causal captions.
- Compute/cost: paired runtime/memory figure and a scenario table; separate measured compute from unmeasured financial assumptions.
- Use appendix/GitHub for full class matrices, extra examples, audits, environment dumps and logs. Retain all original results.

## Scope control and quality

Keep blur, noise and calibration in the plan as requested, with one well-defined intervention and two severities each. No new datasets, broad new training programme or exhaustive corruption grid is assumed. Check frozen protocols and model inputs before inference. Reuse a baseline only if its model, selection, preprocessing and execution controls are demonstrably compatible.

The supplied marking matrix (Academic Year 2024–25 onwards) gives technical work/continuous assessment 25%, dissertation 60% and poster 15%. Within dissertation assessment it weights presentation 1, problem/context 2, technical narrative 3 and discussion/conclusion 1. It groups A1–A5 together; it does not supply an A1-specific guarantee. Prioritize the student's technical understanding, critical evaluation and figures that support the narrative. Ethical/societal context, limitations and professional records also need coverage.

This project demonstrates software integration and controlled empirical evaluation using established models. It does not claim a new detector architecture, novel training method, physical robot deployment or experimentally determined optimum physical sensor configuration.
