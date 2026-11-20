"""TOWER's frozen encoder-frame OpenWAM policy on the XPolicyLab lifecycle."""
from copy import deepcopy
from pathlib import Path

import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import get_robot_action_dim_info, validate_raw_rgb_images_only

from .adapter import CAMERAS, V3OpenWAMPolicy, finite, validate_anchor


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = dict(model_cfg)
        if not validate_raw_rgb_images_only(model_cfg.get("raw_rgb_images_only", True)):
            raise ValueError("OpenWAM_TOWER refuses raw_rgb_images_only: false")
        if model_cfg.get("protocol", "ws") != "ws":
            raise ValueError("OpenWAM_TOWER is served over the ws protocol only")
        if model_cfg.get("env_cfg_type") != "nero_dual":
            raise ValueError("OpenWAM_TOWER requires env_cfg_type=nero_dual")
        if int(model_cfg.get("seed", 42)) != 42:
            raise ValueError("The qualified TOWER inference seed is fixed at 42")
        dimensions = get_robot_action_dim_info("nero_dual")
        if dimensions != {"arm_dim": [7, 7], "ee_dim": [1, 1]}:
            raise ValueError("Unexpected Nero dimensions")
        root = resolve_checkpoint_root(
            model_cfg, Path(__file__).parent / "checkpoints",
            explicit_keys=("package_path", "checkpoint_path"),
        )
        package = root / "package.json" if root.is_dir() else root
        self.policy = V3OpenWAMPolicy(package, device=model_cfg.get("device", "cuda:0"))
        if model_cfg.get("action_type", "eef_delta") != self.policy.mode:
            raise ValueError("Configured action type differs from the checkpoint package")
        self.reset()

    def get_identity(self):
        return dict(self.policy.identity)

    def prepare_case(self, case_meta=None):
        # Episode/environment seeds are independent of the checkpoint's fixed noise seed.
        if case_meta and case_meta.get("inference_seed", 42) != 42:
            raise ValueError("The qualified TOWER inference seed is fixed at 42")

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        if not obs_list:
            raise ValueError("An observation batch cannot be empty")
        batch = {}
        for index, obs in enumerate(obs_list):
            if obs.get("coordinate_frame") != "encoder_arm_base":
                raise ValueError("Expected compensated encoder_arm_base observations")
            state = obs["state"]
            env_idx = int(obs.get("env_idx", index))
            if env_idx in batch:
                raise ValueError("Duplicate environment index")
            images = {}
            for name in CAMERAS:
                pixels = np.asarray(obs["vision"][name]["color"])
                if pixels.ndim != 3 or pixels.shape[-1] != 3 or pixels.dtype != np.uint8:
                    raise ValueError(f"{name} must be RGB uint8 HxWx3")
                images[name] = pixels.copy()
            prompt = obs.get("instruction", obs.get("instructions"))
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("A nonempty instruction is required")
            batch[env_idx] = dict(
                images=images, prompt=prompt,
                joint_encoder=finite(state["joint_encoder"], (2, 7), "joint_encoder").copy(),
                eef_encoder=validate_anchor(state["eef_encoder"]),
                gripper_width_m=finite(state["gripper_width_m"], (2,), "gripper_width_m").copy(),
            )
        self._batch = batch
        self._predictions = {}

    def get_action(self):
        if len(self._batch) != 1:
            raise ValueError("get_action requires one stored observation")
        return self.get_action_batch()[0]

    def get_action_batch(self, env_idx_list=None):
        indices = list(self._batch) if env_idx_list is None else list(env_idx_list)
        if not indices or len(set(indices)) != len(indices):
            raise ValueError("Expected distinct active environment indices")
        if any(int(i) not in self._batch for i in indices):
            raise ValueError("Missing observation for an active environment")
        chunks = []
        # Deliberately serialized: each call uses seed 42 and its own chunk anchor,
        # exactly as the archived HTTP server. Tensor batching is not qualified.
        for index in indices:
            index = int(index)
            prediction = self.policy.predict(self._batch[index])
            self._predictions[index] = prediction
            chunks.append(prediction_to_chunk(prediction))
        return chunks

    def predict_with_evidence(self, observation):
        """Return executed rows and the full horizon in one locked WS CALL."""
        self.update_obs(observation)
        actions = self.get_action()
        return {"actions": actions, "prediction": self.get_prediction()}

    def get_prediction(self, env_idx=None):
        """Full horizon for parity/audit; never performs another inference."""
        index = next(iter(self._batch)) if env_idx is None else int(env_idx)
        return deepcopy(self._predictions[index])

    def reset(self):
        self._batch = {}
        self._predictions = {}


def prediction_to_chunk(prediction):
    """Execute the first 16 of 32 endpoints without losing their frame tags."""
    chunk = []
    for t in range(prediction["execute_steps"]):
        action = {key: deepcopy(prediction[key]) for key in (
            "mode", "coordinate_frame", "units", "control_hz", "execute_steps",
            "applied_j8", "j8_actuator_mask", "identity",
        )}
        action["horizon"] = len(prediction["native_actions"])
        action["native_action"] = np.asarray(prediction["native_actions"][t], dtype=np.float64)
        for arm, name in enumerate(("left", "right")):
            action[f"{name}_ee_joint_state"] = np.asarray(
                [prediction["gripper_width_m"][t][arm]], dtype=np.float64
            )
            if prediction["mode"] == "joint_absolute":
                action[f"{name}_arm_joint_state"] = np.asarray(
                    prediction["joint_encoder_targets"][t][arm], dtype=np.float64
                )
            else:
                # Explicit TOWER extension: absolute TCP transform, not XPL's
                # world-frame pose or a step-relative delta.
                action[f"{name}_ee_transform"] = np.asarray(
                    prediction["eef_encoder_targets"][t][arm], dtype=np.float64
                )
                action["chunk_anchor_encoder"] = np.asarray(prediction["chunk_anchor_encoder"])
        chunk.append(action)
    return chunk
