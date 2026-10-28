"""Preserve public derived evidence from a pinned commit to a verified directory.

Standard-library CPU utility; no model inference or dataset download.
Usage: python scripts/preserve_supervisor_followup.py --commit FULL_SHA --destination /mounted/path
The destination must already exist. For Drive, mount Drive before invoking this script.
"""
import argparse
import hashlib
import json
import re
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

REPO = "abjain182002/multimodal-perception"
PREFIXES = ("results/nuscenes/bevfusion/case_analysis/20261028/",
            "results/supervisor_followup/20261028/",
            "experiments/robot_arm_pick_place/20261022/")
EXTRA = {"docs/report/CASE_COVERAGE_AND_STRUCTURE_2026-10-28.md",
         "docs/report/ILLUMINATION_WEATHER_DISCUSSION_2026-10-28.md",
         "docs/report/ROBOT_ARM_PILOT_2026-10-22.md",
         "docs/report/FOLLOWUP_PRESERVATION_2026-10-28.md",
         "scripts/build_supervisor_followup.py", "scripts/preserve_supervisor_followup.py"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git_blob(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "multimodal-perception-evidence-preservation"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def verify_zip(path):
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        manifest = json.loads(archive.read("preservation_manifest.json"))
        assert set(archive.namelist()) == set(manifest["files"]) | {"preservation_manifest.json"}
        for name, expected in manifest["files"].items():
            data = archive.read(name)
            assert len(data) == expected["bytes"] and sha(data) == expected["sha256"], name
            assert git_blob(data) == expected["git_blob_sha"], name
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    assert re.fullmatch(r"[0-9a-f]{40}", args.commit), "Supply a complete 40-character commit SHA"
    assert args.destination.is_absolute() and args.destination.is_dir(), "Mount/select an existing absolute destination directory first"
    tree = json.loads(fetch(f"https://api.github.com/repos/{REPO}/git/trees/{args.commit}?recursive=1"))
    assert not tree.get("truncated"), "Incomplete repository inventory"
    selected = sorted([x for x in tree["tree"] if x["type"] == "blob" and
                       (x["path"].startswith(PREFIXES) or x["path"] in EXTRA)], key=lambda x: x["path"])
    assert EXTRA <= {x["path"] for x in selected}, "This commit predates the completed follow-up package"
    assert all(any(x["path"].startswith(p) for x in selected) for p in PREFIXES)
    filename = f"supervisor_followup_public_evidence_{args.commit[:12]}.zip"
    destination = args.destination / filename
    receipt_path = args.destination / f"{filename}.receipt.json"
    assert not destination.exists() and not receipt_path.exists(), "Versioned output already exists; preserve it and choose another directory"
    with tempfile.TemporaryDirectory(prefix="supervisor_followup_") as tmp:
        local = Path(tmp) / filename
        manifest = {"repository": REPO, "commit": args.commit,
                    "scope": "Public derived case/arm evidence and synthesis; excludes separately held raw sensors and full predictions.", "files": {}}
        with zipfile.ZipFile(local, "w", zipfile.ZIP_DEFLATED) as archive:
            for index, item in enumerate(selected, 1):
                name = item["path"]
                data = fetch(f"https://raw.githubusercontent.com/{REPO}/{args.commit}/{name}")
                assert len(data) == item["size"] and git_blob(data) == item["sha"], name
                archive.writestr(name, data)
                manifest["files"][name] = {"bytes": len(data), "sha256": sha(data), "git_blob_sha": item["sha"]}
                if index % 10 == 0 or index == len(selected):
                    print(f"VERIFIED PUBLIC FILES: {index}/{len(selected)}", flush=True)
            archive.writestr("preservation_manifest.json", json.dumps(manifest, indent=2) + "\n")
        verify_zip(local)
        expected_zip_sha = sha(local.read_bytes())
        # Exclusive creation preserves any file created after the preceding inventory check.
        with local.open("rb") as source, destination.open("xb") as target:
            shutil.copyfileobj(source, target)
        assert sha(destination.read_bytes()) == expected_zip_sha, "Destination copy hash mismatch"
        copied = verify_zip(destination)
        assert copied == manifest
        receipt = {"status": "PUBLIC_FOLLOWUP_ARCHIVE_AND_DESTINATION_COPY_VERIFIED",
                   "commit": args.commit, "archive": filename, "bytes": destination.stat().st_size,
                   "sha256": expected_zip_sha, "verified_payload_members": len(selected),
                   "destination": str(destination), "scope": manifest["scope"]}
        with receipt_path.open("x", encoding="utf-8") as f:
            f.write(json.dumps(receipt, indent=2) + "\n")
        assert json.loads(receipt_path.read_text()) == receipt
        print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
