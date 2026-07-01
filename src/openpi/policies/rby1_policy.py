import dataclasses

import einops
import numpy as np

from openpi import transforms

ACTION_DIM = 16


def make_rby1_example() -> dict:
    return {
        "observation/front_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/left_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/right_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
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
class RBY1Inputs(transforms.DataTransformFn):
    """Inputs for the RBY1 policy.

    Expected action/state layout:
    - dims 0:7: right arm joints
    - dims 7:14: left arm joints
    - dim 14: right gripper
    - dim 15: left gripper
    """

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/front_image"])
        right_image = _parse_image(data["observation/right_image"])
        left_image = _parse_image(data["observation/left_image"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_image,
                "right_wrist_0_rgb": right_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class RBY1Outputs(transforms.DataTransformFn):
    """Outputs for the RBY1 policy."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :ACTION_DIM])}
