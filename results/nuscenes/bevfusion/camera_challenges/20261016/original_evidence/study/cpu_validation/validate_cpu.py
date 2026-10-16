
from pathlib import Path
import hashlib
import json
import pickle
import cv2
import numpy as np
from corruptions import corrupt_images, corrupt_calibration

local = Path("/content/camera_challenges600")
out = local / "cpu_validation"

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def array_hash(array):
    return hashlib.sha256(
        np.ascontiguousarray(array).tobytes()
    ).hexdigest()

assert np.__version__ == "1.26.4"
assert cv2.__version__ == "4.11.0"
assert sha256(local / "protocol.json") == (
    "ab01b793d13d7b2b06dee318cd99ee3d3c39b6d7f04206538052d90c54533d31"
)
protocol = json.loads((local / "protocol.json").read_text())
conditions = {c["name"]: c for c in protocol["conditions"]}
tokens = (local / "sample_tokens.txt").read_text().splitlines()

print("TESTING IMAGE TRANSFORMATIONS...", flush=True)
generator = np.random.Generator(np.random.PCG64(20261002))
images = generator.uniform(0, 255, (6, 3, 256, 704)).astype(np.float32)
original_hash = array_hash(images)
outputs = {}

for name, condition in conditions.items():
    changed = corrupt_images(images, condition, tokens[0])
    assert changed.shape == images.shape
    assert changed.dtype == np.float32
    assert np.isfinite(changed).all()
    assert changed.min() >= -1e-4 and changed.max() <= 255.0001
    assert array_hash(images) == original_hash

    if condition["family"] in ("clean", "calibration"):
        assert np.array_equal(changed, images)
    else:
        assert not np.array_equal(changed, images)

    outputs[name] = changed
    print("IMAGE CHECK PASS:", name, flush=True)

# Blur preserves a constant image and does not mix views/channels.
constant = np.full_like(images, 128)
for name in ("blur_sigma1", "blur_sigma2"):
    blurred = corrupt_images(constant, conditions[name], tokens[0])
    np.testing.assert_allclose(blurred, constant, rtol=0, atol=1e-3)

impulse = np.zeros_like(images)
impulse[0, 0, 128, 352] = 255
blur1 = corrupt_images(impulse, conditions["blur_sigma1"], tokens[0])
blur2 = corrupt_images(impulse, conditions["blur_sigma2"], tokens[0])
assert 0 < blur2[0, 0, 128, 352] < blur1[0, 0, 128, 352] < 255
assert np.count_nonzero(blur1[1:]) == 0
assert np.count_nonzero(blur1[0, 1:]) == 0

# Noise repeats exactly, changes with token, and shares its base draw.
again = corrupt_images(images, conditions["noise_std10"], tokens[0])
assert np.array_equal(again, outputs["noise_std10"])
different = corrupt_images(images, conditions["noise_std10"], tokens[1])
assert not np.array_equal(different, again)
assert np.all(
    np.abs(outputs["noise_std25"] - images) + 1e-4
    >= np.abs(outputs["noise_std10"] - images)
)

image_checks = {
    "shape_and_float32": True,
    "original_input_unchanged": True,
    "constant_image_blur_preserved": True,
    "blur_does_not_mix_views_or_channels": True,
    "noise_repeatability": True,
    "noise_token_specific": True,
    "shared_noise_realization_across_severities": True,
    "fixture_input_sha256": original_hash,
    "fixture_output_sha256": {
        name: array_hash(value) for name, value in outputs.items()
    },
}
del images, outputs, constant, impulse, blur1, blur2, again, different

print("\nCHECKING ALL ANNOTATION CALIBRATIONS...", flush=True)
ann = local / "nuscenes_camera_availability600_infos_val.pkl"
assert sha256(ann) == (
    "ff6dd3c63e8c3de23b9754947db326d0549bb376271cb7e235a3d54b0c365c44"
)
with ann.open("rb") as stream:
    rows = pickle.load(stream)["data_list"]
assert [row["token"] for row in rows] == tokens
maximum_inverse_error = 0.0
maximum_rotation_error = 0.0
tested = 0

for index, row in enumerate(rows):
    cameras = list(row["images"].items())
    assert len(cameras) == 6
    intrinsic, l2c, c2l = [], [], []

    # Match the pinned BEVFusion loader's construction.
    for channel, camera in cameras:
        assert channel.startswith("CAM_")
        k = np.eye(4, dtype=np.float32)
        raw_k = np.asarray(camera["cam2img"], dtype=np.float32)
        assert raw_k.shape == (3, 3)
        k[:3, :3] = raw_k
        transform = np.asarray(camera["lidar2cam"], dtype=np.float32)
        assert transform.shape == (4, 4)
        inverse = np.eye(4)
        inverse[:3, :3] = transform[:3, :3].T
        inverse[:3, 3:4] = (
            -transform[:3, :3].T @ transform[:3, 3:4]
        )
        intrinsic.append(k)
        l2c.append(transform)
        c2l.append(inverse)

    intrinsic = np.stack(intrinsic)
    l2c = np.stack(l2c)
    c2l = np.stack(c2l)
    metadata = {
        "cam2img": intrinsic,
        "lidar2cam": l2c,
        "cam2lidar": c2l,
        "lidar2img": intrinsic @ l2c,
        "img_aug_matrix": np.repeat(np.eye(4)[None], 6, axis=0),
    }
    before = {key: array_hash(value) for key, value in metadata.items()}

    zero = corrupt_calibration(metadata, 0)
    assert all(np.array_equal(zero[key], value)
               for key, value in metadata.items())

    for angle in (0, 1, 3):
        changed = corrupt_calibration(metadata, angle)
        np.testing.assert_array_equal(changed["cam2img"], intrinsic)
        np.testing.assert_array_equal(
            changed["img_aug_matrix"], metadata["img_aug_matrix"]
        )
        np.testing.assert_allclose(
            changed["lidar2img"],
            intrinsic @ changed["lidar2cam"],
            rtol=1e-6, atol=1e-5,
        )
        product = (
            changed["lidar2cam"].astype(np.float64)
            @ changed["cam2lidar"].astype(np.float64)
        )
        error = float(np.max(np.abs(product - np.eye(4))))
        maximum_inverse_error = max(maximum_inverse_error, error)
        assert error < 5e-5, (row["token"], angle, error)

        rotation = changed["cam2lidar"][:, :3, :3].astype(np.float64)
        rotation_error = float(np.max(np.abs(
            rotation @ rotation.transpose(0, 2, 1) - np.eye(3)
        )))
        maximum_rotation_error = max(maximum_rotation_error, rotation_error)
        assert rotation_error < 1e-5
        np.testing.assert_allclose(
            np.linalg.det(rotation), 1, rtol=0, atol=1e-5
        )

        if angle:
            theta = np.deg2rad(angle)
            delta = np.eye(4)
            delta[:3, :3] = [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
            np.testing.assert_allclose(
                changed["cam2lidar"], delta @ c2l,
                rtol=1e-7, atol=1e-7,
            )
            assert not np.array_equal(changed["cam2lidar"], c2l)
        tested += 6

    assert all(array_hash(value) == before[key]
               for key, value in metadata.items())
    if (index + 1) % 100 == 0:
        print(f"CALIBRATION CHECKS: {index + 1}/600 frames", flush=True)

audit = {
    "status": "CAMERA_CHALLENGES_CPU_TRANSFORMS_PASS",
    "protocol_sha256": sha256(local / "protocol.json"),
    "annotation_sha256": sha256(ann),
    "module_sha256": sha256(out / "corruptions.py"),
    "validation_script_sha256": sha256(Path(__file__)),
    "numpy": np.__version__,
    "opencv": cv2.__version__,
    "image_checks": image_checks,
    "calibration_frames": len(rows),
    "camera_transform_checks_including_zero_angle": tested,
    "maximum_inverse_error": maximum_inverse_error,
    "maximum_rotation_orthogonality_error": maximum_rotation_error,
    "scope": (
        "Synthetic image fixtures and all 600 annotation calibrations "
        "checked on CPU. Actual packed-image/model smoke checks, "
        "GPU inference and accuracy evaluation remain pending."
    ),
}
(out / "cpu_validation_audit.json").write_text(
    json.dumps(audit, indent=2) + "\n"
)
print(json.dumps(audit, indent=2), flush=True)
print("\nCPU IMAGE AND CALIBRATION VALIDATION PASS", flush=True)
