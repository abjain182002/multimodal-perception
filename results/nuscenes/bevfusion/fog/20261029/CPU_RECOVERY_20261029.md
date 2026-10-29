# CPU evaluation recovery: NumPy GT audit serialization

The supplied Colab log (`Pasted markdown(20261029-160217).md`) shows all eight prediction archives verified, metadata restored/derived, SDK initialization and filtering completed, then failure before the first metric evaluation. `json.dumps(frozen_gt.serialize())` rejected an ndarray. The failed attempt is not a completed accuracy study.

The log reports preserved failure ZIP SHA256 `9f31a529f557188aa109e5dfc08812bd988c7c395327568ffa1291761e3e2cdd`, 33,248 bytes, 17 payloads, run `20261029T155657310519Z`. The ZIP itself was not attached or independently reviewed in this recovery; these are log-reported identifiers.

`gt_fingerprint` converts NumPy arrays/scalars only within JSON encoding of the audit representation. It retains valid undefined velocities, stable key ordering, and all ground-truth values. It does not mutate SDK boxes, change filtering, prediction scores, configuration, or metric calculation. Both pre/post evaluation fingerprints now use this helper. The regression check reproduced the original exception with the official SDK 1.2.0 EvalBox/EvalBoxes/DetectionBox class definitions, verified a stable corrected fingerprint, retained NaN velocity and empty frames, and detected a changed ego translation. This is a box serialization regression check, not real-dataset evaluation.

The launcher accepts optional `--reuse-metadata ROOT`. It still verifies all eight saved prediction archives and the weather reference archive. For an existing metadata root, every file is freshly hashed against the pinned weather reference: unchanged files retain original byte counts and hashes; sample_data/ego_pose use the accepted evaluation-only hashes. A mismatch stops execution. If the path no longer exists, normal restoration/derivation runs. Each attempt has a new output directory and Drive evidence ZIP; prior failure evidence is retained.

For this interrupted runtime, ROOT is `/content/fog_cpu_evaluation_20261029/20261029T155657310519Z/metadata_evaluation_only`.

All eight GPU runs remain complete. CPU accuracy metrics, final plots and interpretation still await successful execution and evidence review. No GPU rerun or additional weather experiment is required by this fix.
