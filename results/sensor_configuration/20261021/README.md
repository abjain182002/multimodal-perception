# Sensor-configuration assessment — 21 October 2026

The [assessment](../../../docs/SENSOR_CONFIGURATION_ASSESSMENT_2026-10-21.md) addresses single cameras, one LiDAR, camera–LiDAR combinations and physically separate multiple LiDARs. The [CSV](configurations.csv) records 11 configurations and their evidence status; the [source register](source_register.json) records inspected versions and [decision record](decision_record.json) distinguishes recommendations from executed experiments.

## Findings

- One camera plus one LiDAR is already represented by the KITTI CLOCs work. Its mixed three-seed results and small-data/source-recovery limitations must accompany this evidence.
- A single-camera 3D baseline remains missing. Pinned ImVoxelNet code and a reachable KITTI checkpoint provide a concrete first preflight candidate; no model load, GPU memory fit or inference has been checked today.
- Default nuScenes FCOS3D processes separate images but merges six camera views for a sample. It must not be presented as a one-camera physical system.
- Two- and five-camera availability results retain six computational slots and do not measure the costs of purpose-built reduced-camera systems.
- Waymo Perception supplies separate LiDARs, and OpenPCDet supports their geometric merge. Its inspected model zoo does not distribute Waymo weights. No ready data/checkpoint/resource path was established for this project; the proposed two-/three-/five-LiDAR study is deferred.

The source assessment and project recommendation are complete. New camera-only/multi-LiDAR experiments, supplier prices and supervisor acceptance of scope are not completed by this note. The [report draft](../../../docs/report/SENSOR_CONFIGURATION_DISCUSSION_2026-10-21.md) supplies one compact table and a discussion paragraph; no extra figure is required.

## Verification scope

Current GitHub evidence was read at `5f5c3931afffda42abf381170be6592eec005d4e`. External source blobs were inspected at recorded commits. The ImVoxelNet checkpoint returned HTTP 200 with content length 126,018,456 bytes; this was a header check, not a full checkpoint hash. The BEVDepth release API lists the two-keyframe checkpoint as an uploaded 307,451,655-byte asset; those bytes were not downloaded. Original predictions and metrics were not rerun or modified.

Supporting files have byte sizes and SHA-256 values in [manifest.json](manifest.json). Git publication is checked by comparing uploaded blob identities and reading back the resulting tree.
