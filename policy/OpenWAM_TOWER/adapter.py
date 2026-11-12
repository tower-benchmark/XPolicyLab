"""No actuator conversion: outputs are absolute ENCODER coordinates.

The simulator controller must perform encoder-to-physical conversion exactly once.
Each call owns its EEF anchor; no environment state or action queue is shared.
"""

from __future__ import annotations
import importlib
import importlib.metadata
import platform
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image
from .package import digest_json, sha256, verify_package


class PolicyOutputError(ValueError):
    """The model returned an unusable action, distinct from malformed input."""


CAMERAS = ("right_front_camera", "left_back_camera", "left_wrist_camera", "back_wrist_camera")


def finite(value, shape, label):
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != shape or not np.isfinite(arr).all():
        raise ValueError(f"{label} must be finite {shape}; got {arr.shape}")
    return arr


def rot6d_matrix(value):
    value = np.asarray(value, dtype=np.float64)
    if value.shape[-1] != 6 or not np.isfinite(value).all():
        raise ValueError("Invalid rotation vector")
    first, second = value[..., :3].copy(), value[..., 3:].copy()
    norm = np.linalg.norm(first, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("Degenerate first rotation column")
    first /= norm
    second -= np.sum(first * second, axis=-1, keepdims=True) * first
    norm = np.linalg.norm(second, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("Degenerate second rotation column")
    second /= norm
    return np.stack((first, second, np.cross(first, second)), axis=-1)


def validate_anchor(value):
    anchor = finite(value, (2, 4, 4), "eef_encoder")
    rotation = anchor[:, :3, :3]
    if (
        not np.allclose(anchor[:, 3], [0, 0, 0, 1], atol=1e-6)
        or not np.allclose(rotation.transpose(0, 2, 1) @ rotation, np.eye(3), atol=1e-5)
        or not np.allclose(np.linalg.det(rotation), 1, atol=1e-5)
    ):
        raise ValueError("EEF anchor must be a proper rigid transform")
    return anchor.copy()


def raw_state(mode, observation, open_width, observed_gripper_policy=None):
    widths = finite(observation["gripper_width_m"], (2,), "gripper_width_m")
    minimum = -1e-6
    if observed_gripper_policy is not None:
        accepted_versions = {
            "sim-observed-gripper-closed-undershoot-v1": -0.0005,
            "sim-observed-gripper-closed-undershoot-v2": -0.001,
        }
        version = observed_gripper_policy["version"]
        if version not in accepted_versions:
            raise ValueError("Unsupported observation policy")
        minimum = float(observed_gripper_policy["accepted_minimum_m"])
        if minimum != accepted_versions[version] or observed_gripper_policy["negative_observation_mapping_m"] != 0.0:
            raise ValueError("Unexpected observation policy bounds")
    if np.any(widths < minimum):
        raise ValueError("Negative gripper aperture")
    openness = np.clip(widths / open_width, 0, 1)
    if mode == "joint_absolute":
        joints = finite(observation["joint_encoder"], (2, 7), "joint_encoder")
        return np.concatenate((joints[0], openness[:1], joints[1], openness[1:], [0, 0])).astype(
            np.float32
        )
    anchor = validate_anchor(observation["eef_encoder"])
    parts = []
    for arm in range(2):
        parts.extend(
            (anchor[arm, :3, 3], anchor[arm, :3, 0], anchor[arm, :3, 1], openness[arm : arm + 1])
        )
    return np.concatenate((*parts, [0, 0])).astype(np.float32)


def decode_actions(mode, native, observation, open_width):
    width = 18 if mode == "joint_absolute" else 22
    actions = finite(native, (32, width), "native actions")
    result = dict(
        mode=mode,
        coordinate_frame="encoder_arm_base",
        units={"joint": "rad", "translation": "m", "gripper_width": "m", "predicted_j8": "mm"},
        native_actions=actions.tolist(),
        predicted_j8_mm=actions[:, -2:].tolist(),
        applied_j8="hold_fixed",
        j8_actuator_mask=[False, False],
        control_hz=10,
        execute_steps=16,
    )
    if mode == "joint_absolute":
        result["joint_encoder_targets"] = np.stack(
            (actions[:, :7], actions[:, 8:15]), axis=1
        ).tolist()
        grip = actions[:, [7, 15]]
    else:
        anchor = validate_anchor(observation["eef_encoder"])
        targets = np.broadcast_to(np.eye(4), (32, 2, 4, 4)).copy()
        for arm, offset in enumerate((0, 10)):
            targets[:, arm, :3, 3] = anchor[arm, :3, 3] + actions[:, offset : offset + 3]
            targets[:, arm, :3, :3] = (
                rot6d_matrix(actions[:, offset + 3 : offset + 9]) @ anchor[arm, :3, :3]
            )
        result["eef_encoder_targets"] = targets.tolist()
        result["chunk_anchor_encoder"] = anchor.tolist()
        grip = actions[:, [9, 19]]
    result["gripper_width_m"] = (np.clip(grip, 0, 1) * open_width).tolist()
    result["gripper_projection_count"] = int(np.count_nonzero((grip < 0) | (grip > 1)))
    return result


def install_source(receipt):
    # Freeze exact native SDPA kernels; do not auto-select optional approximate kernels.
    os.environ["WAM_ATTENTION_IMPL"] = "sdpa"
    os.environ["DIFFSYNTH_ATTENTION_IMPLEMENTATION"] = "torch"
    source = Path(receipt["source_snapshot"]).resolve()
    loaded = sys.modules.get("openwam")
    if loaded and not Path(loaded.__file__).resolve().is_relative_to(source):
        raise RuntimeError(
            "A different OpenWAM snapshot is already loaded; use one process per checkpoint"
        )
    sys.path.insert(0, str(source / "third_party"))
    sys.path.insert(0, str(source))


def compose_images(images, cfg):
    # Exact frozen training helpers, including exterior order and crop behavior.
    reader = importlib.import_module("openwam.dataloader.jenga_lerobot")
    multiview = importlib.import_module("openwam.dataloader.transforms.multiview")
    frames = {}
    for name in CAMERAS:
        value = images[name]
        if isinstance(value, Image.Image):
            frames[name] = value.convert("RGB")
        else:
            arr = np.asarray(value)
            if arr.ndim != 3 or arr.shape[2] != 3 or arr.dtype != np.uint8:
                raise ValueError(f"{name} must be RGB uint8 HxWx3")
            frames[name] = Image.fromarray(arr)
    frames["ego_camera"] = reader._concat_exterior(
        frames["right_front_camera"], frames["left_back_camera"]
    )
    return multiview.assemble_multiview_layout(
        frames,
        camera_layout=list(cfg.dataloader.camera_layout),
        out_h=int(cfg.dataloader.height),
        out_w=int(cfg.dataloader.width),
    )


def normalize_absolute_state(state, cfg, checkpoint_dir):
    from openwam.dataloader.utils.normalization import load_stats_file, apply_normalization
    from openwam.dataloader.utils.unify_action import UNIFY_DIM, map_to_unify, parse_unify_spec
    stats = load_stats_file(str(Path(checkpoint_dir)/"normalization_stats.npy"), action_mode="eef_state", normalize_mode="min-max", dim=22)
    raw = np.asarray(state, dtype=np.float32)
    normalized = apply_normalization(raw, stats, "min-max").astype(np.float32)
    return map_to_unify(normalized, parse_unify_spec(cfg.dataloader.unify_action_map, UNIFY_DIM), UNIFY_DIM)[0]

class SeparateStateNormalizer:
    def __init__(self, action_normalizer, cfg, checkpoint_dir):
        self.action = action_normalizer
        self.cfg, self.checkpoint_dir = cfg, checkpoint_dir
    def normalize(self, state):
        return normalize_absolute_state(state, self.cfg, self.checkpoint_dir)
    def unnormalize(self, actions):
        return self.action.unnormalize(actions)

class V3OpenWAMPolicy:
    def __init__(self, package_receipt, device="cuda:0"):
        receipt = (
            json.loads(Path(package_receipt).read_text())
            if isinstance(package_receipt, (str, Path))
            else package_receipt
        )
        self.receipt = verify_package(receipt)
        install_source(self.receipt)
        from omegaconf import OmegaConf
        from openwam.deploy.model_loader import load_from_checkpoint_dir
        from openwam.deploy.engine import JointInferenceEngine
        import torch

        torch.manual_seed(0)
        checkpoint = Path(receipt["checkpoint"])
        self.cfg, self.architecture = load_from_checkpoint_dir(
            str(checkpoint.parent), device=device, ckpt_name=checkpoint.name
        )
        # Strict snapshot loader rejects missing/unexpected tensors. Check numeric integrity
        # separately: a valid safetensors envelope and hash do not imply finite weights.
        for name, parameter in self.architecture.named_parameters():
            if not bool(torch.isfinite(parameter).all()):
                raise ValueError(f"Nonfinite checkpoint tensor: {name}")
        # Explicit frozen baseline; approximation caches disabled pending parity qualification.
        self.cfg = OmegaConf.merge(
            self.cfg,
            {
                "inference": {
                    "denoise_steps": 10,
                    "denoise_mode": "sync",
                    "num_frames": 33,
                    "video_num_frames": 9,
                    "height": 384,
                    "width": 320,
                },
                "optimization": {
                    "decode_video": False,
                    "dit_cache": {"enabled": False},
                    "compile": {"enabled": False},
                    "prompt_embed_cache": {"enabled": True, "maxsize": 32},
                },
            },
        )
        if self.cfg.dataloader.get("state_representation") == "absolute_eef_separate_stats":
            self.architecture.normalizer = SeparateStateNormalizer(self.architecture.normalizer, self.cfg, checkpoint.parent)
        self.engine = JointInferenceEngine(self.cfg, architecture=self.architecture)
        self.mode = receipt["mode"]
        stats = np.load(checkpoint.parent / "normalization_stats.npy", allow_pickle=True).item()
        self.open_width = finite(
            stats["joint" if self.mode == "joint_absolute" else "eef"]["gripper_open_width_m"],
            (2,),
            "open_width",
        )
        if np.any(self.open_width <= 0):
            raise ValueError("Invalid open-width calibration")
        self.identity = dict(
            package_digest=receipt["package_digest"],
            checkpoint_sha256=receipt["checkpoint_sha256"],
            mode=self.mode,
            seed=42,
            denoise_steps=10,
            compile=False,
            dit_cache=False,
            attention_backend="torch_sdpa",
            horizon=32,
            execute_steps=16,
            control_hz=10,
            dtype="bfloat16",
            torch=torch.__version__,
            python=platform.python_version(),
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(torch.device(device))
            if str(device).startswith("cuda")
            else None,
            packages={
                name: importlib.metadata.version(name)
                for name in (
                    "numpy",
                    "pillow",
                    "transformers",
                    "tokenizers",
                    "safetensors",
                    "omegaconf",
                    "diffusers",
                    "accelerate",
                )
            },
            adapter_sha256=sha256(Path(__file__)),
            source_manifest_digest=digest_json(receipt["source_sha256"]),
        )
        self.identity["inference_config_digest"] = digest_json(self.identity)

    def predict(self, observation):
        # Copy anchor before a potentially long model call; caller mutation cannot reanchor.
        observation = dict(observation)
        if self.mode == "eef_delta":
            observation["eef_encoder"] = validate_anchor(observation["eef_encoder"])
        state = raw_state(self.mode, observation, self.open_width, self.receipt["action_contract"].get("observed_gripper_policy"))
        image = compose_images(observation["images"], self.cfg)
        start = time.monotonic()
        result = self.engine.generate(
            dict(
                prompt=str(observation["prompt"]),
                first_frame_image=[image],
                proprio=state,
                seed=42,
                num_frames=33,
                video_num_frames=9,
                denoise_steps=10,
                height=384,
                width=320,
            )
        )
        try:
            decoded = decode_actions(self.mode, result["actions"], observation, self.open_width)
        except (ValueError, KeyError) as exc:
            raise PolicyOutputError(str(exc)) from exc
        decoded.update(identity=self.identity, inference_seconds=time.monotonic() - start)
        return decoded
