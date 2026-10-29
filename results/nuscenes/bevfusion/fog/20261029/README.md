# Bounded synthetic fog study — 29 October 2026

## Status

Source review and the CPU preflight/pilot transfer are implemented. Four local standard-library checks cover full-size population validation and a synthetic end-to-end private handoff. Actual source availability, raw-image/point compatibility, corruption timing, GPU smoke and 600-frame inference remain pending. No fog results exist yet.

The preceding [natural-weather CPU study](../../weather_groups/20261028/completed/README.md) is complete. This is a separate controlled synthetic intervention on the previously frozen 600-frame / 150-scene challenge subset, not another natural-weather stratum.

## Study design

The embedded design in [preflight_fog_colab.py](preflight_fog_colab.py) is saved and verified on Drive before input inspection. Its pilot rule is fixed: first token in the unchanged ordered challenge token list, all six camera files and all ten listed LiDAR input files. Never substitute another frame based on availability or outcomes. Historical sweeps can live under `samples/LIDAR_TOP/`; a path prefix or count cannot identify the current scan. The full model annotation/metadata will resolve that later.

Planned conditions are unperturbed plus fog severities 1/2/3 for each pipeline: **two fresh baseline runs and six fog runs**. Checkpoints, detector source commit, input population, calibration, GT and evaluator stay fixed. Original unperturbed scenes may already contain natural rain/night; unperturbed does not mean clear weather. This compares two existing separately pretrained pipelines and does not isolate the camera component causally.

## Published method and implementation boundary

Source: [MultiCorrupt, official author repository](https://github.com/ika-rwth-aachen/MultiCorrupt/tree/39652e35d10dd7c2aed0b71df8c8c7effe9e6349), pinned commit `39652e35d10dd7c2aed0b71df8c8c7effe9e6349`. Preserve its MIT licence with the supplied code. The preflight downloads four source files, three lookup tables and the licence, verifying their exact Git blob hashes. Assets are not executed/deserialised in this stage.

- LiDAR source `simulate_fog` applies hard-target attenuation and soft fog responses using lookup tables. Severity alpha/beta pairs are (0.02, 0.008), (0.03, 0.008), (0.06, 0.05); converter calls use noise=10, default noise variant v1 and no gain. The converter loads raw float32 Nx5 scans, also corrupts all preceding sweeps when enabled, then stores float32 output. Apply this before scan-coordinate transformations, sweep aggregation, voxelisation and fusion depth construction. Preserve the fifth raw feature and subsequent loader time-lag construction.
- Camera source `fog` combines fractal haze with a spatial distance heuristic, using beta 0.0005/0.001/0.002. The converter decodes raw-resolution BGR images, converts to RGB, then applies the effect before output encoding. It is not a depth-calibrated atmospheric reconstruction.
- Camera and LiDAR severity indices are benchmark settings, not a shared physical visibility distance. Report a synthetic benchmark fog challenge rather than verified real-fog robustness or a physically calibrated joint atmosphere.
- The upstream code uses legacy `np.float_` and `pickle5`, global RNG state and per-pixel/per-point loops. These are compatibility, reproducibility and preparation-time issues to test. Do not silently change arithmetic, coordinate frames, noise handling or intensities for speed. Any accelerated/adapted implementation must be checked against the pinned reference and documented before full inference.

Exact RNG mapping/coupling, image encoding, any acceleration and implementation acceptance checks will be frozen in a follow-on implementation protocol after compatibility tests. The population, pilot-selection rule and intended published severity levels are fixed now. This distinction prevents calling an unvalidated implementation a completed frozen experiment.

## CPU preflight procedure

Use the standard-library script with Drive mounted; CPU is sufficient. It:

1. Preserves the design on Drive, downloads the pinned existing token list and sensor inventory, and verifies the same 600 tokens / 150 scenes / 9,600 unique referenced files.
2. Selects the first frozen token without accessing predictions or GT outcomes.
3. Downloads/verifies the pinned upstream assets without executing them.
4. Checks known extracted sensor roots and a bounded catalogue of project ZIP files (at most 32 ZIPs; project folder and relevant descendants only). Reads central directories for filename availability; does not expand nested TARs or rehash multi-GB archives. Reports skipped/unexpanded sources explicitly.
5. Recovers available pilot files from extracted data or direct ZIP members. ZIP reads check CRC, and recovered bytes get SHA-256 hashes. Filename availability is not full content verification; all complete study inputs must pass their preserved content checks before inference.
6. Records installed package metadata and `nvidia-smi` output when available. No package installation, torch/model import, GPU inference, corruption or environment changes.
7. Produces a manifest-verified private ZIP and hash-verified Drive copy, whether the pilot is ready or missing files. Upload it for source-equivalence testing and the next restoration/runtime step.

Arguments: `--base` preserved project directory, `--work` local work parent, repeatable `--sensor-root` for an existing extracted input root. No need to enable GPU for this step. The pilot input ZIP contains raw sensor data and must remain private; only derived checks/figures will be published.

## Checks and next gates

Run `python check_preflight.py`. Four checks cover complete populations, historical keyframes under `samples/`, malformed sensor counts, safe/ambiguous ZIP indexing, bounded project catalogue, and a synthetic 600-frame / 150-scene transfer with 16 recovered pilot files, pinned opaque assets, no deserialisation and verified output members. These are handoff checks, not fog arithmetic validation.

Next: CPU source-equivalence and determinism tests, real pilot decode/feature checks and timing; final implementation protocol; verified full input restoration; GPU clean/fog smoke; then the eight bounded inference runs and unchanged CPU evaluator. Snow and natural disasters remain outside this measured extension unless explicitly added later.
