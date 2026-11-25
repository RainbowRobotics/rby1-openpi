"""RB-Y1 ego-centric end-effector I/O for the JIMTOF delta / anchored experiments.

The training data are the joint datasets (16-D: right arm 7, left arm 7, right gripper, left gripper; state is
measured, action is commanded). Per arm, the arm FK from `link_torso_5` to `ee_<side>` (RB-Y1 M v1.2 URDF,
rby1-sdk 300a585; A/M v1.1 and v1.2 have the same arm chain) turns joints into hand poses T, and the H action rows
are re-expressed in the hand frame (ego-centric, so torso height or table height does not enter):

    anchored: row k = T(q_t)^-1 T(a_{t+k})                                  every row relative to the measured hand now
    delta:    row 0 = T(q_t)^-1 T(a_t),  row k = T(a_{t+k-1})^-1 T(a_{t+k})  each 30 fps step relative to the previous one

Row layout (14): [R p(3) m, R rotvec(3) rad, R gripper cmd, L p(3), L rotvec(3), L gripper cmd]. Gripper commands
stay absolute (JIMTOF scale: 0.5 open, 0.0 closed).

State given to the model: the 2 measured gripper openings (same scale). No pose, no joints, no history.

Serving: send the 2 gripper openings as `observation/state` (a full 16-D state also works; only 14:16 is read). The
policy returns the 14-D rows; the client turns them into hand targets with `compose_targets` semantics:
    anchored: T_target_k = T_meas · T(row_k)          delta: T_target_k = T_meas · T(row_0) · ... · T(row_k)
"""

from __future__ import annotations

import dataclasses
import functools
import pathlib
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from openpi import transforms
from openpi.policies import rby1_policy

EGO_ACTION_DIM = 14
EGO_STATE_DIM = 2
EGO_MODES = ("anchored", "delta")
DEFAULT_URDF = pathlib.Path(__file__).parent / "rby1_models" / "rby1m_model_v1.2.urdf"
# column layout of the 16-D joint data and of the 14-D ego rows
_ARMS = {"right": (slice(0, 7), 14, 0), "left": (slice(7, 14), 15, 7)}


def _rpy_matrix(rpy) -> np.ndarray:
    return Rotation.from_euler("xyz", rpy).as_matrix()  # URDF fixed-axis roll-pitch-yaw: Rz(y) Ry(p) Rx(r)


class ArmFK:
    """Vectorized FK of one arm, `base_link` -> `tip_link`, straight from the URDF joint origins and axes."""

    def __init__(self, urdf_path: str | pathlib.Path, base_link: str, tip_link: str):
        root = ET.parse(str(urdf_path)).getroot()
        by_child = {}
        for j in root.findall("joint"):
            by_child[j.find("child").get("link")] = j
        path, link = [], tip_link
        while link != base_link:
            if link not in by_child:
                raise KeyError(f"{tip_link} is not below {base_link} in {urdf_path}")
            j = by_child[link]
            path.append(j)
            link = j.find("parent").get("link")
        self.steps = []  # (fixed origin 4x4, unit axis or None)
        self.joint_names = []
        for j in reversed(path):
            o = j.find("origin")
            T = np.eye(4)
            if o is not None:
                T[:3, :3] = _rpy_matrix([float(v) for v in o.get("rpy", "0 0 0").split()])
                T[:3, 3] = [float(v) for v in o.get("xyz", "0 0 0").split()]
            kind = j.get("type")
            if kind in ("revolute", "continuous"):
                ax = np.array([float(v) for v in j.find("axis").get("xyz").split()])
                self.steps.append((T, ax / np.linalg.norm(ax)))
                self.joint_names.append(j.get("name"))
            elif kind == "fixed":
                self.steps.append((T, None))
            else:
                raise ValueError(f"unsupported joint type {kind} ({j.get('name')})")

    def __call__(self, q: np.ndarray) -> np.ndarray:
        """q: (N, 7) joint angles (rad) in chain order -> (N, 4, 4) poses of the tip in the base link."""
        q = np.asarray(q, dtype=np.float64)
        n = q.shape[0]
        T = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
        i = 0
        for origin, axis in self.steps:
            T = T @ origin
            if axis is not None:
                Tj = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
                Tj[:, :3, :3] = Rotation.from_rotvec(np.outer(q[:, i], axis)).as_matrix()
                T = T @ Tj
                i += 1
        return T


@functools.lru_cache(maxsize=4)
def arm_fks(urdf_path: str = str(DEFAULT_URDF)) -> dict[str, ArmFK]:
    fks = {side: ArmFK(urdf_path, "link_torso_5", f"ee_{side}") for side in _ARMS}
    for side, fk in fks.items():
        expected = [f"{side}_arm_{i}" for i in range(7)]
        if fk.joint_names != expected:
            raise ValueError(f"{urdf_path}: {side} chain joints {fk.joint_names} != {expected}")
    return fks


def inv(T: np.ndarray) -> np.ndarray:
    Ti = np.zeros_like(T)
    Rt = np.swapaxes(T[..., :3, :3], -1, -2)
    Ti[..., :3, :3] = Rt
    Ti[..., :3, 3] = -np.einsum("...ij,...j->...i", Rt, T[..., :3, 3])
    Ti[..., 3, 3] = 1.0
    return Ti


def to_matrix(row6: np.ndarray) -> np.ndarray:
    """(..., 6) [p, rotvec] -> (..., 4, 4)."""
    row6 = np.asarray(row6, dtype=np.float64)
    T = np.zeros((*row6.shape[:-1], 4, 4))
    T[..., :3, :3] = Rotation.from_rotvec(row6.reshape(-1, 6)[:, 3:]).as_matrix().reshape(*row6.shape[:-1], 3, 3)
    T[..., :3, 3] = row6[..., :3]
    T[..., 3, 3] = 1.0
    return T


def ego_rows(T_meas: np.ndarray, T_cmd: np.ndarray, mode: str) -> np.ndarray:
    """One arm: measured hand pose (4, 4) and commanded chunk (H, 4, 4) -> (H, 6) hand-frame rows."""
    if mode == "anchored":
        rel = inv(T_meas)[None] @ T_cmd
    elif mode == "delta":
        prev = np.concatenate([T_meas[None], T_cmd[:-1]], axis=0)
        rel = inv(prev) @ T_cmd
    else:
        raise ValueError(f"mode must be one of {EGO_MODES}, got {mode!r}")
    return np.concatenate([rel[:, :3, 3], Rotation.from_matrix(rel[:, :3, :3]).as_rotvec()], axis=-1)


def compose_targets(T_meas: np.ndarray, rows6: np.ndarray, mode: str) -> np.ndarray:
    """Inverse of `ego_rows`: (4, 4) measured pose + (H, 6) rows -> (H, 4, 4) targets (what the client does)."""
    Tr = to_matrix(rows6)
    if mode == "anchored":
        return T_meas[None] @ Tr
    if mode == "delta":
        out, T = [], T_meas
        for k in range(Tr.shape[0]):
            T = T @ Tr[k]
            out.append(T)
        return np.stack(out)
    raise ValueError(f"mode must be one of {EGO_MODES}, got {mode!r}")


def joint_chunk_to_ego(state: np.ndarray, actions: np.ndarray, mode: str, urdf_path: str = str(DEFAULT_URDF)) -> np.ndarray:
    """16-D measured state + (H, 16) commanded joint chunk -> (H, 14) ego rows (both arms + gripper commands)."""
    state = np.asarray(state, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    if state.shape[-1] != 16 or actions.ndim != 2 or actions.shape[-1] != 16:
        raise ValueError(f"expected 16-D state and (H, 16) actions, got {state.shape} and {actions.shape}")
    fks = arm_fks(urdf_path)
    out = np.empty((actions.shape[0], EGO_ACTION_DIM))
    for side, (jc, gc, oc) in _ARMS.items():
        T_meas = fks[side](state[None, jc])[0]
        T_cmd = fks[side](actions[:, jc])
        out[:, oc : oc + 6] = ego_rows(T_meas, T_cmd, mode)
        out[:, oc + 6] = actions[:, gc]
    return out


def gripper_state(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)
    if state.shape[-1] == 16:
        return state[..., 14:16]
    if state.shape[-1] == EGO_STATE_DIM:
        return state
    raise ValueError(f"expected a 16-D joint state or the 2 gripper openings, got shape {state.shape}")


@dataclasses.dataclass(frozen=True)
class RBY1EgoInputs(transforms.DataTransformFn):
    """Images/prompt as RBY1Inputs; state -> 2 measured gripper openings; actions -> ego rows (training only)."""

    mode: str = "anchored"
    urdf_path: str = str(DEFAULT_URDF)

    def __post_init__(self):
        if self.mode not in EGO_MODES:
            raise ValueError(f"mode must be one of {EGO_MODES}, got {self.mode!r}")

    def __call__(self, data: dict) -> dict:
        inputs = rby1_policy.RBY1Inputs()(data)
        raw_state = inputs["state"]
        if "actions" in inputs:
            inputs["actions"] = joint_chunk_to_ego(raw_state, inputs["actions"], self.mode, self.urdf_path).astype(
                np.float32
            )
        inputs["state"] = gripper_state(raw_state)
        return inputs


@dataclasses.dataclass(frozen=True)
class RBY1EgoOutputs(transforms.DataTransformFn):
    """Return the 14-D ego rows (the model pads actions to its action_dim)."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., :EGO_ACTION_DIM]}
