# Description-defined weather groups: CPU extension

## Status

Stage 1 preparation is implemented and checked locally on synthetic metadata. Actual 150-scene group counts, description review and CPU subgroup evaluation are pending user-run Colab access to preserved metadata/predictions. No measured weather-group result is claimed yet. Fog and optional snow are separate future experiments.

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

## Stage 2: pending description review and evaluator preparation

Review the complete inventory before reading new subgroup metrics. Then verify full saved prediction hashes, recreate the pinned CPU evaluator and GT filtering, and compare full-split results to the saved reference before interpreting group metrics. Report pooled group mAP/NDS, per-class AP and filtered GT counts. Explicitly disclose absent or sparse classes and limitations of macro averages. Do not derive group AP by averaging frame or scene AP. Any uncertainty computation must resample scenes, preserve paired pipeline membership, and record its method/seed/repetitions before execution; scene intervals still do not remove shared-log or dataset selection limitations.

Interpret pipeline differences within each group first. Differences between natural groups confound weather with location, class composition, distance, visibility, point support and other scene properties. They do not isolate rain/night causality or the causal contribution of cameras.

## Local verification

Run `python check_preparation.py`. Checks cover word boundaries/negative mention flags; rejection of incomplete scenes and duplicate metadata; timestamp ordering; a synthetic 6,019-frame / 150-scene ZIP/TGZ recovery; pinned token-file verification; manifest and Drive-copy hash checks; prediction non-parsing; and the restored-metadata alternative. Real data remain to be executed and reviewed.

## Dissertation boundary

Measured natural strata and later synthetic fog must remain separate. Snow, glare, wind, severe storms and flooding can be critically covered through primary literature, limitations and targeted future validation. Their inclusion in writing does not establish detector performance under them.
