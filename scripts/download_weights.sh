#!/usr/bin/env bash
# Download the ASH weights from Hugging Face into ./ASH, the layout config/ash_inference.yaml expects.
#
# Fetches the base models and start state (assets/) plus ONE checkpoint, checkpoint-5 by default
# (about 15 GB in total). Pick another with CHECKPOINT=N (1-10).
#
# Usage: scripts/download_weights.sh
set -euo pipefail
cd "$(dirname "$0")/.."

REPO="Toolbox-HQ/ASH"
DEST="ASH"
CHECKPOINT="${CHECKPOINT:-5}"

if command -v hf >/dev/null 2>&1; then
    HF=(hf)
elif command -v uvx >/dev/null 2>&1; then
    HF=(uvx --from "huggingface_hub[cli]" hf)
else
    echo "Need the Hugging Face CLI: install uv (https://docs.astral.sh/uv/) or 'pip install huggingface_hub[cli]'." >&2
    exit 1
fi

"${HF[@]}" download "${REPO}" \
    --include "assets/*" \
    --include "checkpoints/checkpoint-${CHECKPOINT}/*" \
    --include "rom/*" \
    --local-dir "${DEST}"

echo
echo "Weights are in ./${DEST}/assets and ./${DEST}/checkpoints/checkpoint-${CHECKPOINT}."
echo "Now copy your Pokémon Emerald ROM to ./${DEST}/rom/rom.gba (it is not distributed)."
if [[ "${CHECKPOINT}" != "5" ]]; then
    echo "config/ash_inference.yaml points at checkpoint-5: update load_path, idm_path and objective_load_path to checkpoint-${CHECKPOINT}."
fi
