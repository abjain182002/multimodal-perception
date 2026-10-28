# Description-defined weather groups: CPU extension

## Status

Stages 1 and 2 have passed on real preserved data. The preparation ZIP is 184,365 bytes, SHA-256 `d9aa5377247df83162cbfd8126638e4ded39e6459f036e378cf4f35cc47ad1a0`; all five payloads pass checks. All 150 descriptions were reviewed before subgroup metrics and literal groups were retained. See [preparation receipt](preparation_import_receipt.json), [description review](description_review.json) and [scene inventory](scene_inventory.csv). The [completed CPU analysis](completed/README.md) now preserves the final source ZIP, exact tables, audits and coverage-aware figure; the [report discussion](../../../../../docs/report/NATURAL_WEATHER_GROUPS_2026-10-28.md) separates aggregate gains, absent classes, pedestrian difficulty and log correlation. Fog and optional snow remain separate future experiments.

| Description group | Scenes | Frames | Logs |
|---|---:|---:|---:|
| Neither mentioned | 111 | 4,449 | 15 |
| Rain only mentioned | 24 | 968 | 2 |
| Night only mentioned | 12 | 482 | 1 |
| Night and rain mentioned | 3 | 120 | 1 |

There are **18 unique logs** overall. All 15 night-mentioned scenes share one log, which spans both night strata. Group log counts therefore cannot be summed as independent acquisition counts. The description review found no negations or alternative condition wording requiring a group amendment; this does not verify weather in individual images.

## Question and scope

How do the two existing pretrained pipelines compare within scene-description strata of the already evaluated full-validation population (6,019 samples / 150 scenes)? This is a post-hoc extension: prior aggregate and selected-case outcomes have been inspected. Group membership is frozen before new subgroup metrics are read, not before all project outcomes were known.

Use four exhaustive, disjoint literal-description groups: neither condition mentioned, rain only mentioned, night only mentioned, and both mentioned. Word-boundary patterns are case insensitive: `rain`, `rainy`, `raining`; `night`, `nighttime`. These are scene-level mentions, not measured frame-level weather. No mention is not proof of daylight or dry weather. Review all descriptions for negation, synonyms and transitional conditions before evaluation. Keep the literal inventory and document any sensitivity-analysis amendment before examining group metrics.

The complete protocol is embedded in [prepare_weather_groups.py](prepare_weather_groups.py) and is saved on Drive before metadata grouping. Rain/night marginal summaries overlap and must not be treated as independent groups. Counts by scene, frame and acquisition log are reported; even different scenes can share a log.

## Stage 1: CPU preparation

Run the pinned script with Drive mounted. It uses only the Python standard library:

1. Save the protocol locally and verify its Drive copy.
2. Download the three existing validation token lists at commit `3c606fa1cea1b7112442f85f318993705ddfdacf`; verify Git blob hashes and exact 6,019-token union.
3. Read `scene.json` and `sample.json` from restored metadata or stream them from `nuscenes_validation_parts_01_02_verified.zip` / `metadata/v1.0-trainval_meta.tgz`. Verify both table SHA-256 hashes against the preserved [metadata restoration audit](../../full_validation/replay_20261020/original_evidence/setup/metadata_restoration_audit.json). Do not rehash the entire multi-GB ZIP or extract images.
4. Require exactly 150 complete scenes and reject missing/duplicate samples. Freeze scene descriptions, acquisition-log identities and every sample's group membership.
5. Record saved prediction availability without opening their contents. Check candidate existing CPU environments; no package installation or inference.
6. Save a manifest-verified compact ZIP, verify the Drive copy, and print its upload path.

Optional arguments: `--base` (existing preserved project folder), `--root` (restored nuScenes root), `--work` (local preparation parent). New UTC timestamps prevent overwriting earlier runs. If a hash or population check fails, stop rather than selecting a smaller subset.

## Stage 2: completed; reproduction procedure

Run [run_weather_groups_colab.py](run_weather_groups_colab.py) with `--code-commit` set to its full published commit SHA. The launcher re-verifies the frozen preparation ZIP and preserves the description review on Drive before metric access. It fetches the worker and [resolved package pins](requirements-cpu310.txt) with SHA-256 checks, then creates an isolated Python 3.10.22 environment using uv 0.13.4. Core scientific packages and all four evaluator source hashes match the prior replay; remaining resolved dependencies and the runtime package freeze are preserved.

Only metadata is restored from the original ZIP; each of the 19 files is checked against its preserved size/SHA-256. No images or detector dependencies are installed and the entire multi-GB source ZIP is not rehashed. An **evaluation-only** derivative retains all annotations, temporal annotation links, samples and scenes unchanged, while streaming only validation keyframe `sample_data` rows and their referenced ego poses into smaller tables. This reduces memory use without discarding the annotations needed for velocity calculation. The derivative is not suitable for inference, sweeps or training.

The [worker](evaluate_weather_groups.py) runs each model in a fresh process, verifies its full prediction hash, requires exact official validation membership, and first replays the full split. Both summary and detailed metrics must match the preserved original results within **1e-12**, excluding only `eval_time`. A mismatch stops subgroup analysis. The full-replay check also guards the metadata derivative against changed evaluation behaviour.

Four subgroup evaluations per model select from the official already-filtered GT/predictions, including empty frames and preserving original token order for score ties. Outputs include pooled SDK ten-class mAP/NDS, per-class AP, GT/prediction counts, absent classes, and classes with fewer than 20 GT instances (a disclosure marker, not a validity threshold). Separately labelled represented-class and common-GT-class macro AP provide sensitivity summaries; these do not replace official mAP. Do not derive group AP by averaging scene AP.

Before metrics, the reviewed protocol elects **descriptive point estimates only**, with no confidence intervals or significance tests. One night log and two rain-only logs do not support broad between-log weather generalisation; scene bootstrap would not resolve that limitation. If inferential analysis is later added, document its estimand, clustering assumptions, algorithm and settings before execution.

After both model runs pass, the launcher produces paired CSV/JSON summaries and a compact PNG/PDF figure, copies each completed model's evidence to Drive, and creates a manifest-verified ZIP with a hash-verified Drive copy and receipt. Raw predictions and restored/derived GT metadata are excluded from the evidence ZIP. Original data remain unchanged. A new timestamp is used for each execution; a failure preserves local logs without claiming completion.

Interpret pipeline differences within each group first. Differences between natural groups confound weather with location, class composition, distance, visibility, point support and other scene properties. They do not isolate rain/night causality or the causal contribution of cameras.

## Local verification

Run `python check_preparation.py`. Checks cover word boundaries/negative mention flags; rejection of incomplete scenes and duplicate metadata; timestamp ordering; a synthetic 6,019-frame / 150-scene ZIP/TGZ recovery; pinned token-file verification; manifest and Drive-copy hash checks; prediction non-parsing; and the restored-metadata alternative. These three checks passed before real preparation; the actual preparation and description review have now passed as recorded above.

Stage 2: run `check_evaluation.py` in the pinned Python 3.10.22 environment. Four checks cover exact core package versions and evaluator source hashes; recursive replay comparison including NaN handling and changed-value rejection; empty-frame/order retention; and synthetic full-versus-derived metadata evaluation through the actual SDK. The synthetic fixture includes a GT velocity link to an annotation outside the selected sensor population. Checks also exercise ZIP/TGZ restoration, rejection of corrupted restored tables, absent-class reporting and figure generation. The synthetic figure was visually inspected locally and is not published as a real result. Local checks and the real 6,019-frame subgroup execution have passed; see completed evidence above.

## Dissertation boundary

Measured natural strata and later synthetic fog must remain separate. Snow, glare, wind, severe storms and flooding can be critically covered through primary literature, limitations and targeted future validation. Their inclusion in writing does not establish detector performance under them.
