#!/usr/bin/env bash
set -euo pipefail
# Arg 8 accepts a conda name, conda/uv environment directory, or Python executable.
bench_name=$1 task_name=$2 ckpt_name=$3 env_cfg_type=$4 action_type=$5
seed=$6 policy_gpu_id=$7 policy_env=$8 port=$9 host=${10:-localhost}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [[ -x "$policy_env/bin/python" ]]; then
  policy_python="$policy_env/bin/python"
elif [[ -f "$policy_env" && -x "$policy_env" ]]; then
  policy_python="$policy_env"
else
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$policy_env"
  policy_python="python"
fi
export PYTHONPATH="$XPL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec env CUDA_VISIBLE_DEVICES="$policy_gpu_id" "$policy_python" "$XPL_ROOT/setup_policy_server.py" \
  --config_path "$SCRIPT_DIR/deploy.yml" --overrides \
  port="$port" host="$host" bench_name="$bench_name" task_name="$task_name" \
  ckpt_name="$ckpt_name" env_cfg_type="$env_cfg_type" action_type="$action_type" seed="$seed"
