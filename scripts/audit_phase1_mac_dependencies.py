#!/usr/bin/env python3
"""Read-only Mac inventory for the pinned Phase 1 project dependencies."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from scripts.package_clocs_train_export_batch import manifest_members


PINNED_REPOS = {
    "mmdetection3d": ("mmdetection3d", "fe25f7a51d36e3702f961e198894580d83c4387b"),
    "OpenPCDet": ("OpenPCDet", "233f849829b6ac19afb8af8837a0246890908755"),
    "BEVFusion": ("BEVFusion", "fb4384ca05cae05be548dd96a0f940da70263cdb"),
    "Azitt": ("Lidar-CameraFusion-3DobjectDetection",
              "b59b49d2e915cd3b4a3b816b9c228e811601e241"),
    "MultiCorrupt": ("MultiCorrupt", "39652e3"),
}

REQUIRED_PATHS = {
    "kitti_train_label_dir": "datasets/KITTI/training/label_2",
    "kitti_train_calib_dir": "datasets/KITTI/training/calib",
    "kitti_train_image_dir": "datasets/KITTI/training/image_2",
    "kitti_train_velodyne_dir": "datasets/KITTI/training/velodyne",
    "second_train_split": "external/CLOCs/second/data/ImageSets/train.txt",
    "second_validation_split": "external/CLOCs/second/data/ImageSets/val.txt",
    "mmdet_pointpillars_config":
        "external/mmdetection3d/configs/pointpillars/"
        "pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py",
    "azitt_entrypoint": "external/Lidar-CameraFusion-3DobjectDetection/main.py",
    "azitt_predictor": "external/Lidar-CameraFusion-3DobjectDetection/inference/predictor.py",
    "azitt_pvrcnn_config":
        "external/Lidar-CameraFusion-3DobjectDetection/detection/pvrcnn_config.json",
    "azitt_expected_weights":
        "external/Lidar-CameraFusion-3DobjectDetection/"
        "lidar-camera-models/pv_rcnn_8369.pth",
    "openpcdet_kitti_pvrcnn_plus_config":
        "external/OpenPCDet/tools/cfgs/kitti_models/"
        "pv_rcnn_plusplus_reproduced_by_community.yaml",
    "bevfusion_nuscenes_config":
        "external/BEVFusion/configs/bevfusion/bevf_pp_2x8_1x_nusc.py",
}


def git_revision(repo):
    """Return a revision only when the directory itself is the Git root."""
    if not repo.is_dir():
        return None
    top = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--show-toplevel"],
        capture_output=True, text=True, check=False)
    if top.returncode or Path(top.stdout.strip()).resolve() != repo.resolve():
        return None
    rev = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False)
    return rev.stdout.strip() if rev.returncode == 0 else None


def audit_dependencies(root, manifest):
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    _, members = manifest_members(manifest)
    dataset = root / "datasets/KITTI"
    training = dataset / "training"
    source_bytes = sum((training / name.removeprefix("training/")).stat().st_size
                       for name in members)
    repos = {}
    for name, (relative, expected) in PINNED_REPOS.items():
        directory = root / "external" / relative
        actual = git_revision(directory)
        repos[name] = {
            "relative_path": f"external/{relative}", "directory_present": directory.is_dir(),
            "expected_commit": expected, "actual_commit": actual,
            "commit_matches": bool(actual and actual.startswith(expected)),
        }
    return {
        "schema_version": "phase1_mac_dependency_inventory_v1",
        "evidence_type": "path_and_revision_inventory_only",
        "project_root_name": root.name,
        "pilot_input_members": len(members),
        "pilot_uncompressed_input_bytes": source_bytes,
        "project_disk_free_bytes": shutil.disk_usage(root).free,
        "required_paths": {
            name: {"relative_path": path, "present": (root / path).exists()}
            for name, path in REQUIRED_PATHS.items()
        },
        "external_repositories": repos,
        "nuscenes_candidates": {
            path: {
                "directory_present": (root / path).is_dir(),
                "mini_metadata_present": (root / path / "v1.0-mini").is_dir(),
                "trainval_metadata_present": (root / path / "v1.0-trainval").is_dir(),
            } for path in (
                "datasets/nuscenes", "datasets/nuScenes",
                "external/BEVFusion/data/nuscenes",
            )
        },
        "limitations": [
            "Path/revision presence does not prove model or checkpoint inference.",
            "nuScenes metadata presence alone does not prove complete sensor files or annotations.",
            "The Azitt entrypoint's CUDA model and PV-RCNN++ weights must be checked separately.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Home of datasets/ and external/")
    parser.add_argument("--manifest", default="results/kitti/clocs/training_exports/clocs_train_export_batch_manifest.json")
    parser.add_argument("--output", default="results/audits/environment/phase1_mac_dependency_inventory.json")
    args = parser.parse_args()
    report = audit_dependencies(args.root, args.manifest)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

