# OpenWAM_TOWER

Eval-only TOWER embodiment of OpenWAM, served through XPolicyLab's WebSocket
server. The existing `OpenWAM` RoboDojo policy is unchanged. No checkpoint is
included or downloaded. Training and data conversion are outside this adapter.

## Installation and deployment

Use Linux x86_64 and Python 3.12 (reference interpreter: 3.12.3). From the
XPolicyLab checkout, with `uv` installed, create an isolated environment and
install the pinned serving stack:

```bash
uv venv --python python3.12 --seed .venv-policy
source .venv-policy/bin/activate
python -m pip install --require-hashes -r policy/OpenWAM_TOWER/requirements-policy.txt
python -m pip check
```

The lock pins torch 2.11.0+cu130 and torchvision 0.26.0+cu130 from the official
CUDA 13.0 wheel index, with Triton 3.6.0, transport and training-source serving
dependencies. It requires an NVIDIA GPU with bf16 support and an
[NVIDIA R580 or newer driver](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
(CUDA 13.x minimum driver family). Inference uses bf16 and native torch SDPA.
Reference scores were produced on a different accelerator with this torch
version, so per-scene outcomes may differ between hardware stacks and only
aggregate rates are comparable.

Supply the **exact training source snapshot supplied with the checkpoint** via
the package receipt; the adapter verifies and imports that snapshot directly.
The current upstream OpenWAM tree is not a substitute. The lock includes its
serving dependencies; no separate training-package installation is needed.
Do not install the snapshot's broad training requirements over this environment,
as that can replace the pinned numerical stack.

Copy `embodiments/nero_dual/env_cfg/` into the parent benchmark's `env_cfg/`
(merge the robot registry if one exists). Both runtime and training dimension
registries then declare two seven-joint arms and two one-dimensional grippers.
TOWER provides this parent configuration beside its submodule.

```bash
python setup_policy_server.py --config_path policy/OpenWAM_TOWER/deploy.yml \
  --overrides port=19000 package_path=/path/to/package.json
```

Alternatively, from this policy directory:

```bash
bash setup_eval_policy_server.sh TOWER task checkpoint_run nero_dual eef_delta \
  42 0 /path/to/policy-env 19000 localhost
```

`ckpt_name` follows the shared checkpoint resolver. `package_path` is an explicit
receipt file or a directory containing `package.json`. A receipt identifies a
safetensors checkpoint, tokenizer, config, normalization stats, action contract,
and exact training source manifest. `python -m
XPolicyLab.policy.OpenWAM_TOWER.package --help` describes receipt creation.
Receipt paths are deployment configuration; moving files requires a new receipt
and therefore a new package digest. Weight and source verification is mandatory.
The inherited NumPy normalization sidecar uses pickle and must be trusted.

## Observation and action contract

`env_cfg_type: nero_dual`, `action_type: eef_delta` or `joint_absolute`, matching
the package. Four RGB uint8 cameras are `right_front_camera`, `left_back_camera`,
`left_wrist_camera`, `back_wrist_camera`, under `vision/<name>/color`. The server
owns decoding; the adapter only validates and copies arrays, then uses the exact
training crop and multiview composition. `instruction` is the prompt.

`state` contains `joint_encoder` (2,7) radians, `eef_encoder` (2,4,4) TCP matrices,
and `gripper_width_m` (2,) full aperture in metres. `coordinate_frame` must be
`encoder_arm_base`. Arm index 0 is left; index 1 is back (XPolicyLab's `right`).
**The environment owns J2 compensation:** physical observations are converted to
encoder coordinates before sending; encoder commands are converted back exactly
once by the controller. This policy never applies actuator compensation or IK.
Existing observed-gripper tolerance is read unchanged from the package contract.

The model generates 32 endpoints at 10 Hz and returns the first 16 to execute.
Joint actions use standard `left/right_arm_joint_state` fields. EEF actions use
TOWER-specific `left/right_ee_transform` matrices: absolute encoder TCP targets
reconstructed from **one anchor per chunk** as `p = p_anchor + delta_p` and
`R = delta_R @ R_anchor`, with the archived rot6d decoder. They must not be treated
as generic XPolicyLab world-frame `ee_pose`. Grippers use
`left/right_ee_joint_state` in metres, not normalized openness. Each action carries
mode, frame, units, timing and identity. Predicted J8 is diagnostic only; hold it
fixed. `get_prediction()` exposes the last full 32-step prediction for audits.

Inference noise seed is fixed at 42 on every call. Other configured inference
seeds are rejected. `prepare_case` may carry an independent environment seed;
`reset` clears all observations and anchors. Batch methods accept explicit
`env_idx` values and run independent calls serially, preserving archived seed
and per-observation anchors. This is lifecycle batching, not tensor batching.

## Provenance and limitations

`adapter.py` and `package.py` are mechanically ported from TOWER's archived
November 2026 OpenWAM serving adapter (Apache-2.0). Their numerical behavior,
validation, package identity and digest computation are preserved. The checkpoint
source is external and hash-verified; no internal deployment locations are
embedded here. Full GPU parity must be established in TOWER before claiming
comparability. No new grasp generation, IK bounds, observation tolerances or
hardware protection changes are made by this adapter.

## Evaluation script entry points

The three standard wrappers keep the upstream positional argument convention.
The policy environment argument accepts a conda name, a conda/uv environment
prefix (containing `bin/python`), or a Python executable. The environment-side
wrapper uses the upstream benchmark client and its conda environment convention.
The benchmark must supply the Nero encoder observation/controller contract above;
the generic TOWER CPU backend is only a protocol smoke fixture, not that client.

```bash
# Same machine: bench task checkpoint embodiment mode seed policyGPU envGPU policyEnv envEnv
bash eval.sh TOWER task checkpoint_run nero_dual eef_delta 42 0 0 /path/to/policy-env benchmark-env
# Split machine: bench task checkpoint embodiment mode seed envGPU envEnv info port host
bash setup_eval_env_client.sh TOWER task checkpoint_run nero_dual eef_delta 42 0 benchmark-env \
  'ckpt_name=checkpoint_run,action_type=eef_delta' 19000 localhost
```

Set `EVAL_ENV_TYPE=sim` for a benchmark-provided simulator client. The upstream
`debug` client must likewise be configured to supply the TOWER state fields; its
generic joint-only fixture is insufficient. `eval.sh` waits for server readiness
and stops/reaps its owned policy server when the client exits. Provision the
checkpoint receipt through `ckpt_name` or `deploy.yml`; these scripts do not fetch
weights or change inference configuration. This remains an eval-only integration;
training release timing and checkpoint distribution are owner decisions.
