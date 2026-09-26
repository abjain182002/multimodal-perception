#!/usr/bin/env python3
"""Real archive round-trip and tamper detection for the train-only GPU queue."""

import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from scripts.package_clocs_train_export_batch import package_bundle, verify_bundle


class PackageTrainExportBatchTests(unittest.TestCase):
    def test_round_trip_and_tampered_source_or_member(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "kitti"
            frame = "000149"
            contents = {
                "calib": (".txt", b"calibration"),
                "image_2": (".png", b"image"),
                "label_2": (".txt", b"Car"),
                "velodyne": (".bin", b"lidar points"),
            }
            paths, hashes = {}, {}
            for part, (suffix, data) in contents.items():
                path = dataset / part / f"{frame}{suffix}"
                path.parent.mkdir(parents=True)
                path.write_bytes(data)
                paths[part] = f"{part}/{frame}{suffix}"
                hashes[part] = hashlib.sha256(data).hexdigest()
            manifest = root / "manifest.json"
            train = root / "train.txt"
            val = root / "val.txt"
            train.write_text("000149\n")
            val.write_text("000001\n")
            manifest.write_text(json.dumps({
                "schema_version": "clocs_train_export_batch_v1",
                "status": "INPUTS_VERIFIED_EXPORT_PENDING",
                "split": "train",
                "requested_size": 1,
                "selected_ids_sha256": hashlib.sha256(b"000149\n").hexdigest(),
                "train_split_sha256": hashlib.sha256(train.read_bytes()).hexdigest(),
                "validation_split_sha256": hashlib.sha256(val.read_bytes()).hexdigest(),
                "frames": [{
                    "frame_id": frame, "input_files": paths, "input_sha256": hashes,
                }],
            }))
            archive = root / "batch.tar.gz"
            saved = package_bundle(manifest, dataset, archive, train, val)
            self.assertEqual(saved["input_members_verified"], 4)
            self.assertEqual(saved["split_members_verified"], 2)
            checked = verify_bundle(manifest, archive, root / "restored")
            self.assertEqual(saved["archive_sha256"], checked["archive_sha256"])
            for part, (_, data) in contents.items():
                self.assertEqual((root / "restored" / "training" / paths[part]).read_bytes(),
                                 data)
            self.assertEqual((root / "restored/splits/train.txt").read_bytes(),
                             train.read_bytes())
            val.write_text("000149\n")
            with self.assertRaisesRegex(ValueError, "Frozen split SHA-256 mismatch"):
                package_bundle(manifest, dataset, root / "wrong-split.tar.gz",
                               train, val)
            val.write_text("000001\n")
            (dataset / "image_2" / f"{frame}.png").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "KITTI input SHA-256 mismatch"):
                package_bundle(manifest, dataset, root / "bad.tar.gz", train, val)
            with tarfile.open(root / "extra.tar.gz", "w:gz") as tampered:
                other = tarfile.TarInfo("../outside")
                other.size = 4
                tampered.addfile(other, io.BytesIO(b"evil"))
            with self.assertRaisesRegex(ValueError, "Unexpected, duplicate or unsafe"):
                verify_bundle(manifest, root / "extra.tar.gz", root / "unsafe")
            self.assertFalse((root / "unsafe").exists())


if __name__ == "__main__":
    unittest.main()
