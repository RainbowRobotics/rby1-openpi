import json
import pathlib

import numpy as np
import pytest

from openpi.policies import rby1_ego_policy as ego

_REF = pathlib.Path(__file__).parent / "rby1_models" / "rby1_ego_fk_reference.json"
_LIMITS = np.radians([[-180, 180], [-180, 1], [-180, 180], [-150, 1], [-180, 180], [-90, 110], [-155, 155]])


def _random_joints(rng, n, side):
    lim = _LIMITS.copy()
    if side == "left":  # arm_1 range is mirrored on the left arm
        lim[1] = np.radians([-1, 180])
    lo, hi = np.maximum(lim[:, 0], -2.5), np.minimum(lim[:, 1], 2.5)
    return rng.uniform(lo, hi, size=(n, 7))


def _smooth_chunk(rng, H=40):
    """A 16-D measured state and an (H, 16) commanded chunk that moves smoothly away from it."""
    q0 = np.concatenate([_random_joints(rng, 1, "right")[0], _random_joints(rng, 1, "left")[0]])
    steps = rng.normal(scale=0.01, size=(H, 14)).cumsum(axis=0) + rng.normal(scale=0.02, size=14)
    grip = rng.choice([0.0, 0.5], size=(H, 2))
    state = np.concatenate([q0, rng.uniform(0, 0.5, size=2)])
    actions = np.concatenate([q0[None] + steps, grip], axis=1)
    return state, actions


def test_fk_matches_rby1_sdk():
    ref = json.loads(_REF.read_text())
    fks = ego.arm_fks()
    for side in ("right", "left"):
        q = np.array([r["q"] for r in ref[side]])
        T_ref = np.array([r["T"] for r in ref[side]])
        np.testing.assert_allclose(fks[side](q), T_ref, atol=1e-7)  # SDK agrees to ~4e-9 m


@pytest.mark.parametrize("mode", ego.EGO_MODES)
def test_rows_recompose_to_the_commanded_poses(mode):
    rng = np.random.default_rng(0)
    fks = ego.arm_fks()
    for _ in range(5):
        state, actions = _smooth_chunk(rng)
        rows = ego.joint_chunk_to_ego(state, actions, mode)
        assert rows.shape == (40, ego.EGO_ACTION_DIM)
        for side, (jc, gc, oc) in ego._ARMS.items():
            T_meas = fks[side](state[None, jc])[0]
            targets = ego.compose_targets(T_meas, rows[:, oc : oc + 6], mode)
            np.testing.assert_allclose(targets, fks[side](actions[:, jc]), atol=1e-9)
            np.testing.assert_array_equal(rows[:, oc + 6], actions[:, gc])


def test_anchored_and_delta_agree_on_the_first_row():
    rng = np.random.default_rng(1)
    state, actions = _smooth_chunk(rng)
    a = ego.joint_chunk_to_ego(state, actions, "anchored")
    d = ego.joint_chunk_to_ego(state, actions, "delta")
    np.testing.assert_allclose(a[0], d[0], atol=1e-12)


def test_rows_are_ego_centric():
    """Moving the whole arm base (e.g. another torso height) must not change the rows: FK starts at link_torso_5 and
    the rows are relative poses, so a left-multiplied rigid transform cancels."""
    rng = np.random.default_rng(2)
    fks = ego.arm_fks()
    state, actions = _smooth_chunk(rng)
    jc = slice(0, 7)
    T_meas, T_cmd = fks["right"](state[None, jc])[0], fks["right"](actions[:, jc])
    W = ego.to_matrix(np.array([0.1, -0.2, 0.35, 0.3, -0.2, 0.9]))
    for mode in ego.EGO_MODES:
        np.testing.assert_allclose(
            ego.ego_rows(W @ T_meas, W[None] @ T_cmd, mode), ego.ego_rows(T_meas, T_cmd, mode), atol=1e-9
        )


def test_padding_at_the_episode_end():
    """lerobot repeats the last frame past the episode end: delta rows become zero motion, anchored rows constant."""
    rng = np.random.default_rng(3)
    state, actions = _smooth_chunk(rng)
    actions[30:] = actions[29]
    d = ego.joint_chunk_to_ego(state, actions, "delta")
    a = ego.joint_chunk_to_ego(state, actions, "anchored")
    np.testing.assert_allclose(d[30:, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]], 0.0, atol=1e-12)
    np.testing.assert_allclose(a[30:], np.broadcast_to(a[29], a[30:].shape), atol=1e-12)


def _example(state, actions=None):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    data = {
        "observation/front_image": img,
        "observation/left_image": img.transpose(1, 0, 2),
        "observation/right_image": img.transpose(1, 0, 2),
        "observation/state": state,
        "prompt": "do something",
    }
    if actions is not None:
        data["actions"] = actions
    return data


@pytest.mark.parametrize("mode", ego.EGO_MODES)
def test_inputs_training_and_serving(mode):
    rng = np.random.default_rng(4)
    state, actions = _smooth_chunk(rng)
    out = ego.RBY1EgoInputs(mode=mode)(_example(state, actions))
    assert out["state"].shape == (2,)
    np.testing.assert_allclose(out["state"], state[14:16].astype(np.float32))
    assert out["actions"].shape == (40, ego.EGO_ACTION_DIM) and out["actions"].dtype == np.float32
    np.testing.assert_allclose(out["actions"], ego.joint_chunk_to_ego(state, actions, mode), atol=1e-6)
    # serving: no actions; the client may send only the 2 gripper openings or the full 16-D state
    for s in (state[14:16], state):
        served = ego.RBY1EgoInputs(mode=mode)(_example(s))
        assert "actions" not in served
        np.testing.assert_allclose(served["state"], state[14:16].astype(np.float32))
    with pytest.raises(ValueError):
        ego.RBY1EgoInputs(mode=mode)(_example(state[:7]))


def test_outputs_slice_the_padded_actions():
    padded = np.arange(40 * 32, dtype=np.float32).reshape(40, 32)
    np.testing.assert_array_equal(ego.RBY1EgoOutputs()({"actions": padded})["actions"], padded[:, :14])


def test_bad_mode_is_rejected():
    with pytest.raises(ValueError):
        ego.RBY1EgoInputs(mode="absolute")
