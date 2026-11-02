#!/usr/bin/env bash
# Run ASH with a config inside the container built by scripts/build_env.sh.
# One agent runs per visible GPU; choose GPUs with CUDA_VISIBLE_DEVICES.
#
# Usage: scripts/run.sh [config]            (default: config/ash_inference.yaml)
#
# Environment overrides:
#   CONTAINER     container image           (default .cache/containers/ash.sif)
#   RUN_UUID      run name                  (default: a fresh UUID)
#   WANDB_MODE    offline|online|disabled   (default offline; set WANDB_API_KEY for online)
#   MASTER_PORT   torchrun rendezvous port  (default 35332)
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-config/ash_inference.yaml}"
CONTAINER="${CONTAINER:-.cache/containers/ash.sif}"
RUN_UUID="${RUN_UUID:-$(uuidgen 2>/dev/null || date +run_%Y%m%d_%H%M%S)}"

[[ -f "${CONTAINER}" ]]  || { echo "Container not found at ${CONTAINER}. Run scripts/build_env.sh first." >&2; exit 1; }
[[ -f "${CONFIG}" ]]     || { echo "Config not found: ${CONFIG}" >&2; exit 1; }
[[ -d ASH/assets ]]      || { echo "Weights not found in ./ASH. Run scripts/download_weights.sh first." >&2; exit 1; }
[[ -f ASH/rom/rom.gba ]] || { echo "ROM not found. Copy your Pokémon Emerald ROM to ASH/rom/rom.gba." >&2; exit 1; }

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    NUM_GPUS=$(tr ',' '\n' <<< "${CUDA_VISIBLE_DEVICES}" | wc -l)
else
    NUM_GPUS=$(nvidia-smi -L | wc -l)
fi
mkdir -p .cache/tmp .cache/cupy .cache/hf logs

EXTRA_ENV=()
[[ -n "${WANDB_API_KEY:-}" ]] && EXTRA_ENV+=(--env "WANDB_API_KEY=${WANDB_API_KEY}")

echo "Run ${RUN_UUID}: ${NUM_GPUS} agent(s), config ${CONFIG}"
apptainer exec \
    --contain \
    --nv \
    --bind ./.cache/tmp:/tmp \
    --bind ./.cache:/app/.cache \
    --bind ./ASH:/app/ASH \
    --bind ./logs:/app/logs \
    --bind ./config:/app/config \
    --env HF_HOME=/app/.cache/hf \
    --env HF_HUB_OFFLINE=1 \
    --env TRITON_HOME=/app/.cache/tmp \
    --env TRITON_CACHE_DIR=/app/.cache/tmp \
    --env CUPY_CACHE_DIR=/app/.cache/cupy \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env WANDB_MODE="${WANDB_MODE:-offline}" \
    --env PYTHONUNBUFFERED=1 \
    ${EXTRA_ENV[@]+"${EXTRA_ENV[@]}"} \
    "${CONTAINER}" \
    bash -c "cd /app && . .venv/bin/activate && \
        torchrun \
          --nproc_per_node=${NUM_GPUS} \
          --nnodes=1 \
          --node_rank=0 \
          --master_addr=localhost \
          --master_port=${MASTER_PORT:-35332} \
          main.py \
          --config \"${CONFIG}\" \
          --uuid \"${RUN_UUID}\" \
          --seed_rng false"

echo "Run ${RUN_UUID} finished. Outputs are in logs/online/${RUN_UUID} (gameplay videos under query_video/)."
