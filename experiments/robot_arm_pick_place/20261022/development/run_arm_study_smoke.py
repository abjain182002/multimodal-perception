"""CPU rigid-body pick/place pilot; simulator-state feedback, no perception model."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random

import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DT = 1 / 240
PROFILES = {
    "light_component": {"mass": 0.08, "half_extents": [0.025, 0.025, 0.03], "friction": 0.8},
    "heavy_component": {"mass": 0.5, "half_extents": [0.025, 0.025, 0.03], "friction": 0.4},
    "narrow_component": {"mass": 0.12, "half_extents": [0.012, 0.012, 0.04], "friction": 0.5},
    "development": {"mass": 0.2, "half_extents": [0.025, 0.025, 0.03], "friction": 0.6},
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def trial(profile, policy, seed, out, capture=False):
    cfg = PROFILES[profile]
    rng = random.Random(seed)
    offset = [rng.uniform(-0.003, 0.003), rng.uniform(-0.003, 0.003)]
    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(DT)
    p.setPhysicsEngineParameter(numSolverIterations=100, deterministicOverlappingPairs=1)
    p.loadURDF("plane.urdf")
    robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
    home = [0, -0.3, 0, -2.2, 0, 2.0, math.pi / 4]
    for j, q in enumerate(home):
        p.resetJointState(robot, j, q)
    for j in (9, 10):
        p.resetJointState(robot, j, 0.04)
        p.changeDynamics(robot, j, lateralFriction=1.0)
    shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=cfg["half_extents"])
    visual = p.createVisualShape(p.GEOM_BOX, halfExtents=cfg["half_extents"], rgbaColor=[0.1, 0.5, 0.85, 1])
    h = cfg["half_extents"][2]
    obj = p.createMultiBody(cfg["mass"], shape, visual, [0.5 + offset[0], offset[1], h + 0.001])
    p.changeDynamics(obj, -1, lateralFriction=cfg["friction"], restitution=0)
    # Destination marker has visual geometry only, not an attachment/support constraint.
    marker = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.07, 0.07, 0.0005], rgbaColor=[0.1, 0.7, 0.3, 0.8])
    p.createMultiBody(0, -1, marker, [0.5, 0.22, 0.0005])
    orn = p.getQuaternionFromEuler([math.pi, 0, 0])
    start = np.array(p.getLinkState(robot, 11, computeForwardKinematics=True)[4])
    # Grasp link is between finger tips; height offset avoids the ground.
    grasp_z = h + 0.005
    phases = [
        ("approach", [0.5, 0, 0.25], 2.0, 0.04),
        ("descend", [0.5, 0, grasp_z], 2.0, 0.04),
        ("close", [0.5, 0, grasp_z], 1.0, 0.0),
        ("lift", [0.5, 0, 0.25], 2.0, 0.0),
        ("hold", [0.5, 0, 0.25], 1.0, 0.0),
        ("transfer", [0.5, 0.22, 0.25], 2.0, 0.0),
        ("lower", [0.5, 0.22, grasp_z], 2.0, 0.0),
        ("release", [0.5, 0.22, grasp_z], 1.0, 0.04),
        ("retreat", [0.5, 0.22, 0.25], 1.0, 0.04),
        ("settle", [0.5, 0.22, 0.25], 1.0, 0.04),
    ]
    cap = 20.0 if policy == "fixed_high" else 3.0
    trace, forces, hold_z, transfer_z = [], [], [], []
    prev_rel = None
    steps = 0
    peak_height = 0.0
    for phase, target, seconds, width in phases:
        target = np.array(target)
        count = round(seconds / DT)
        for k in range(count):
            u = (k + 1) / count
            u = u * u * (3 - 2 * u)
            desired = start + (target - start) * u
            q = p.calculateInverseKinematics(robot, 11, desired, orn, maxNumIterations=80, residualThreshold=1e-5)
            p.setJointMotorControlArray(robot, list(range(7)), p.POSITION_CONTROL, targetPositions=q[:7], forces=[87, 87, 87, 87, 12, 12, 12])
            for j in (9, 10):
                p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, targetPosition=width, force=cap, maxVelocity=0.05)
            p.stepSimulation()
            steps += 1
            xyz = p.getBasePositionAndOrientation(obj)[0]
            ee = p.getLinkState(robot, 11, computeForwardKinematics=True)[4]
            normals = [sum(c[9] for c in p.getContactPoints(robot, obj, linkIndexA=j)) for j in (9, 10)]
            if phase in ("close", "lift", "hold", "transfer", "lower"):
                forces.append(sum(normals))
            if phase == "hold":
                hold_z.append(xyz[2])
            if phase == "transfer":
                transfer_z.append(xyz[2])
            peak_height = max(peak_height, xyz[2])
            if steps % 12 == 0:
                rel = xyz[2] - ee[2]
                old_cap = cap
                if policy == "slip_feedback" and phase in ("lift", "hold", "transfer"):
                    # Ideal state-derived slip proxy, updated at 20 Hz; no object mass input.
                    if prev_rel is not None and rel - prev_rel < -0.0002:
                        cap = min(20.0, cap + 2.0)
                prev_rel = rel
                trace.append({"time_s": steps * DT, "phase": phase, "object_z_m": xyz[2], "ee_z_m": ee[2], "normal_force_sum_N": sum(normals), "motor_limit_N": old_cap, "next_motor_limit_N": cap})
        start = target
        if capture and phase in ("close", "hold", "transfer", "settle"):
            view = p.computeViewMatrixFromYawPitchRoll([0.4, 0.06, 0.18], 1.05, 45, -27, 0, 2)
            proj = p.computeProjectionMatrixFOV(48, 1.4, 0.01, 3)
            rgba = p.getCameraImage(560, 400, view, proj, renderer=p.ER_TINY_RENDERER)[2]
            Image.fromarray(np.asarray(rgba, dtype=np.uint8).reshape(400, 560, 4)[:, :, :3]).save(out / f"{profile}_{policy}_{phase}.png")
    final, quat = p.getBasePositionAndOrientation(obj)
    vel, ang = p.getBaseVelocity(obj)
    xy_error = math.hypot(final[0] - 0.5, final[1] - 0.22)
    stable_lift = min(hold_z) > h + 0.12 and min(transfer_z) > h + 0.12
    placed = stable_lift and xy_error < 0.03 and abs(final[2] - h) < 0.01 and np.linalg.norm(vel) < 0.02
    result = {"profile": profile, "policy": policy, "seed": seed, "offset_x_m": offset[0], "offset_y_m": offset[1], "stable_lift": bool(stable_lift), "placed": bool(placed), "xy_error_m": xy_error, "peak_object_z_m": peak_height, "peak_normal_sum_N": max(forces), "mean_normal_sum_N": float(np.mean(forces)), "final_motor_limit_N": cap, "final_z_m": final[2], "final_speed_m_s": float(np.linalg.norm(vel))}
    with (out / f"trace_{profile}_{policy}_{seed}.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=trace[0]); w.writeheader(); w.writerows(trace)
    p.disconnect(client)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "evaluate"], default="smoke")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "smoke":
        settings = [("development", "fixed_high", 0)]
    else:
        protocol_path = ROOT / "protocol.json"
        protocol = json.loads(protocol_path.read_text())
        assert protocol["script_sha256"] == digest(__file__), "Script differs from frozen protocol"
        settings = [(name, policy, seed) for name in protocol["profiles"] for seed in protocol["seeds"] for policy in protocol["policies"]]
    results = []
    for name, policy, seed in settings:
        row = trial(name, policy, seed, args.output, capture=(seed == 0))
        results.append(row)
        print(json.dumps(row), flush=True)
    with (args.output / "trials.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=results[0]); w.writeheader(); w.writerows(results)
    audit = {"mode": args.mode, "python": platform.python_version(), "platform": platform.platform(), "pybullet": importlib.metadata.version("pybullet"), "numpy": np.__version__, "script_sha256": digest(__file__), "panda_urdf_sha256": digest(Path(pybullet_data.getDataPath()) / "franka_panda/panda.urdf"), "trials": len(results), "no_object_attachment_constraints": True, "scope": "Rigid-body simulation with ideal simulator-state pose/slip inputs; no detector, tactile sensor, physical damage or hardware validation."}
    if args.mode == "evaluate":
        audit["protocol_sha256"] = digest(ROOT / "protocol.json")
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    main()
