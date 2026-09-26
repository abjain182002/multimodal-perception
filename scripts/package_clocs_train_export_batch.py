#!/usr/bin/env python3
"""Package/verify an exact train-only KITTI pilot for later Colab GPU export.

The trusted Git-tracked manifest controls every file in the archive. Verification
is CPU-only, rejects unexpected members and checks the bytes of all 64 inputs.
"""

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import tarfile
import tempfile
from pathlib import Path

from scripts.export_clocs_dense_train_frame import file_sha256


PARTS = {"calib": ".txt", "image_2": ".png",
         "label_2": ".txt", "velodyne": ".bin"}


def manifest_members(manifest_path):
    raw = Path(manifest_path).read_bytes()
    data = json.loads(raw)
    if (data.get("schema_version") != "clocs_train_export_batch_v1"
            or data.get("status") != "INPUTS_VERIFIED_EXPORT_PENDING"
            or data.get("split") != "train"):
        raise ValueError("Manifest is not a verified train-only export queue")
    frames = data.get("frames")
    if not isinstance(frames, list) or len(frames) != data.get("requested_size"):
        raise ValueError("Manifest frame count is inconsistent")
    members = {}
    ids = []
    for frame in frames:
        frame_id = frame["frame_id"]
        if not isinstance(frame_id, str) or not re.fullmatch(r"[0-9]{6}", frame_id):
            raise ValueError("Invalid KITTI frame identifier")
        ids.append(frame_id)
        if set(frame["input_sha256"]) != set(PARTS) or set(frame["input_files"]) != set(PARTS):
            raise ValueError(f"Incomplete inputs for {frame_id}")
        for part, suffix in PARTS.items():
            expected = f"{part}/{frame_id}{suffix}"
            if frame["input_files"][part] != expected:
                raise ValueError(f"Unexpected source path for {frame_id}: {part}")
            digest = frame["input_sha256"][part]
            if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
                raise ValueError(f"Invalid source SHA-256 for {frame_id}: {part}")
            members["training/" + expected] = digest
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate frame in export manifest")
    selected = hashlib.sha256(("\n".join(ids) + "\n").encode("ascii")).hexdigest()
    if selected != data.get("selected_ids_sha256"):
        raise ValueError("Manifest selected-ID hash mismatch")
    return raw, members


def split_members(trusted_manifest):
    data = json.loads(trusted_manifest)
    output = {}
    for name, key in (("splits/train.txt", "train_split_sha256"),
                      ("splits/val.txt", "validation_split_sha256")):
        digest = data.get(key)
        if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError(f"Invalid split SHA-256: {key}")
        output[name] = digest
    return output


def _tar_info(name, size):
    info = tarfile.TarInfo(name)
    info.size = size
    info.mode = 0o644
    info.mtime = 0
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    return info


def verify_bundle(manifest_path, archive_path, extract_to=None):
    """Verify exact manifest bytes and member hashes before any extraction."""
    trusted, expected = manifest_members(manifest_path)
    splits = split_members(trusted)
    want = {"manifest.json": hashlib.sha256(trusted).hexdigest(),
            **splits, **expected}
    with tarfile.open(archive_path, "r:gz") as bundle:
        observed = {}
        for member in bundle:
            if member.name not in want or member.name in observed or not member.isfile():
                raise ValueError(f"Unexpected, duplicate or unsafe archive member: {member.name}")
            digest = hashlib.sha256()
            source = bundle.extractfile(member)
            if source is None:
                raise ValueError(f"Cannot read archive member: {member.name}")
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
            observed[member.name] = digest.hexdigest()
            if observed[member.name] != want[member.name]:
                raise ValueError(f"Archive input SHA-256 mismatch: {member.name}")
        if set(observed) != set(want):
            raise ValueError(f"Archive member count mismatch: {len(observed)} != {len(want)}")
    if extract_to is not None:
        target = Path(extract_to)
        with tarfile.open(archive_path, "r:gz") as bundle:
            for member in bundle:
                if member.name == "manifest.json":
                    continue
                destination = target / member.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                # Only the exact, verified regular members above are eligible.
                with bundle.extractfile(member) as source, destination.open("wb") as output:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        output.write(block)
        for name, digest in {**splits, **expected}.items():
            if file_sha256(target / name) != digest:
                raise ValueError(f"Restored input SHA-256 mismatch: {name}")
    return {
        "status": "PASS", "num_frames": len(expected) // len(PARTS),
        "input_members_verified": len(expected),
        "split_members_verified": len(splits),
        "manifest_sha256": hashlib.sha256(trusted).hexdigest(),
        "archive_sha256": file_sha256(archive_path),
        "archive_bytes": Path(archive_path).stat().st_size,
        "extracted_to": str(extract_to) if extract_to is not None else None,
    }


def package_bundle(manifest_path, dataset, output_path, train_split, validation_split):
    trusted, expected = manifest_members(manifest_path)
    dataset = Path(dataset)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    splits = split_members(trusted)
    split_paths = {"splits/train.txt": Path(train_split),
                   "splits/val.txt": Path(validation_split)}
    for name, digest in splits.items():
        if file_sha256(split_paths[name]) != digest:
            raise ValueError(f"Frozen split SHA-256 mismatch: {name}")
    # Check every source before writing; repeat verification on the final tar.
    for name, digest in expected.items():
        source = dataset / name.removeprefix("training/")
        if file_sha256(source) != digest:
            raise ValueError(f"KITTI input SHA-256 mismatch: {source}")
    with tempfile.NamedTemporaryFile(
            prefix=".clocs_export_", suffix=".tar.gz",
            dir=output_path.parent, delete=False) as temporary:
        staged = Path(temporary.name)
    try:
        with staged.open("wb") as raw_file:
            with gzip.GzipFile(fileobj=raw_file, mode="wb", filename="", mtime=0) as zipped:
                with tarfile.open(fileobj=zipped, mode="w") as bundle:
                    bundle.addfile(_tar_info("manifest.json", len(trusted)),
                                   io.BytesIO(trusted))
                    for name, source in split_paths.items():
                        with source.open("rb") as data_file:
                            bundle.addfile(_tar_info(name, source.stat().st_size),
                                           data_file)
                    for name in expected:
                        source = dataset / name.removeprefix("training/")
                        with source.open("rb") as data_file:
                            bundle.addfile(_tar_info(name, source.stat().st_size),
                                           data_file)
        result = verify_bundle(manifest_path, staged)
        os.replace(staged, output_path)
        result["archive_sha256"] = file_sha256(output_path)
        return result
    finally:
        staged.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    pack = sub.add_parser("pack")
    pack.add_argument("--manifest", required=True)
    pack.add_argument("--dataset", required=True)
    pack.add_argument("--train", required=True)
    pack.add_argument("--val", required=True)
    pack.add_argument("--output", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--manifest", required=True)
    verify.add_argument("--archive", required=True)
    verify.add_argument("--extract-to")
    args = parser.parse_args()
    if args.action == "pack":
        result = package_bundle(args.manifest, args.dataset, args.output,
                                args.train, args.val)
    else:
        result = verify_bundle(args.manifest, args.archive, args.extract_to)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
