#!/usr/bin/env bash
# Build the Apptainer container ASH runs in. It holds the Python environment (installed with uv
# from uv.lock), the mGBA emulator library, and a copy of this repository at /app.
#
# Output: .cache/containers/ash.sif (override with CONTAINER=path). Rebuild after changing code.
# If your system needs it, pass extra build flags with APPTAINER_BUILD_ARGS="--fakeroot".
#
# Usage: scripts/build_env.sh
set -euo pipefail
cd "$(dirname "$0")/.."

CONTAINER="${CONTAINER:-.cache/containers/ash.sif}"

command -v apptainer >/dev/null 2>&1 || { echo "apptainer not found: https://apptainer.org/docs/admin/main/installation.html" >&2; exit 1; }
mkdir -p "$(dirname "${CONTAINER}")" .cache/tmp .cache/cupy logs

echo "Building ${CONTAINER} from dconfig/apptainer_run.def (this takes several minutes)..."
# shellcheck disable=SC2086
apptainer build ${APPTAINER_BUILD_ARGS:-} "${CONTAINER}" dconfig/apptainer_run.def
echo "Built ${CONTAINER}"
