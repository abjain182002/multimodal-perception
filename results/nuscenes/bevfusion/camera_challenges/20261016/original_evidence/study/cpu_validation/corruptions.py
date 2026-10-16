
import copy
import hashlib
import cv2
import numpy as np

def corrupt_images(images, condition, token):
    """Float32 (views, channels, height, width), before normalization."""
    source = np.asarray(images)
    assert source.shape == (6, 3, 256, 704), source.shape
    assert np.isfinite(source).all()
    assert source.min() >= 0 and source.max() <= 255
    result = source.astype(np.float32, copy=True)
    family = condition["family"]

    if family in ("clean", "calibration"):
        return result

    if family == "blur":
        sigma = float(condition["sigma_pixels"])
        kernel = int(condition["kernel_size"])
        assert sigma > 0 and kernel > 0 and kernel % 2 == 1
        for view in range(6):
            image = np.ascontiguousarray(result[view].transpose(1, 2, 0))
            blurred = cv2.GaussianBlur(
                image, (kernel, kernel),
                sigmaX=sigma, sigmaY=sigma,
                borderType=cv2.BORDER_REFLECT_101,
            )
            result[view] = blurred.transpose(2, 0, 1)
        return result

    if family == "noise":
        seed = int.from_bytes(
            hashlib.sha256(
                ("20261002:camera_noise:" + token).encode("utf-8")
            ).digest()[:8],
            "big",
        )
        generator = np.random.Generator(np.random.PCG64(seed))
        noise = generator.standard_normal(result.shape).astype(np.float32)
        std = float(condition["standard_deviation_pixel_units"])
        return np.clip(result + std * noise, 0, 255).astype(np.float32)

    raise ValueError(f"Unknown image corruption family: {family}")

def corrupt_calibration(metadata, yaw_degrees):
    """Apply a common LiDAR-frame yaw error to all camera extrinsics."""
    result = copy.deepcopy(metadata)
    if float(yaw_degrees) == 0:
        return result

    angle = np.deg2rad(float(yaw_degrees))
    delta = np.eye(4, dtype=np.float64)
    delta[:3, :3] = [
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0, 0, 1],
    ]

    old_c2l = np.asarray(metadata["cam2lidar"])
    old_l2c = np.asarray(metadata["lidar2cam"])
    old_l2i = np.asarray(metadata["lidar2img"])
    intrinsic = np.asarray(metadata["cam2img"])
    assert all(a.shape == (6, 4, 4) for a in
               (old_c2l, old_l2c, old_l2i, intrinsic))

    changed_c2l = np.stack([
        delta @ matrix.astype(np.float64) for matrix in old_c2l
    ]).astype(old_c2l.dtype)

    changed_l2c = np.stack([
        np.linalg.inv(matrix.astype(np.float64))
        for matrix in changed_c2l
    ]).astype(old_l2c.dtype)

    changed_l2i = np.stack([
        k @ extrinsic for k, extrinsic in zip(intrinsic, changed_l2c)
    ]).astype(old_l2i.dtype)

    result["cam2lidar"] = changed_c2l
    result["lidar2cam"] = changed_l2c
    result["lidar2img"] = changed_l2i
    return result
