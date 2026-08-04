import dataclasses

import einops
import numpy as np

from openpi import transforms

ARM_JOINT_DIM = 7
ACTION_DIM = 8


def make_rby1_one_arm_example() -> dict:
    """Creates a fake single-arm RB-Y1 observation for testing."""
    return {
        "observation/front_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/state": np.random.rand(ACTION_DIM).astype(np.float32),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class RBY1OneArmInputs(transforms.DataTransformFn):
    """Inputs for a single-arm, single-camera RB-Y1 policy.

    Expected action/state layout:
    - dims 0:7: arm joints
    - dim 7: gripper

    The PI model keeps its three standard image slots. The two unavailable
    wrist-camera slots are filled with zeros and disabled through image_mask.
    """

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/front_image"])
        empty_image = np.zeros_like(base_image)

        state = np.asarray(data["observation/state"], dtype=np.float32)
        if state.shape[-1] != ACTION_DIM:
            raise ValueError(
                f"Expected single-arm RB-Y1 state dimension {ACTION_DIM}, "
                f"but received shape {state.shape}."
            )

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": empty_image,
                "right_wrist_0_rgb": empty_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.False_,
            },
        }

        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.shape[-1] != ACTION_DIM:
                raise ValueError(
                    f"Expected single-arm RB-Y1 action dimension {ACTION_DIM}, "
                    f"but received shape {actions.shape}."
                )
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class RBY1OneArmOutputs(transforms.DataTransformFn):
    """Outputs for a single-arm RB-Y1 policy."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        return {"actions": actions[..., :ACTION_DIM]}