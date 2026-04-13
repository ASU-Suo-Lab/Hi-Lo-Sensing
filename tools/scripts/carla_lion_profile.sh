#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ "$#" -lt 1 ]; then
    echo "Usage: bash scripts/carla_lion_profile.sh <Mamba|FLA_GLA> [save_dir] [additional profile_lion.py args...]"
    exit 1
fi

OPERATOR_NAME="$1"
shift

case "${OPERATOR_NAME}" in
    Mamba)
        DEFAULT_SAVE_DIR="../output/lion_profile/lion_mamba_stable"
        FUSED_MAPPING="False"
        ;;
    FLA_GLA)
        DEFAULT_SAVE_DIR="../output/lion_profile/lion_fla_stable"
        FUSED_MAPPING="True"
        ;;
    *)
        echo "Unsupported LION operator: ${OPERATOR_NAME}"
        echo "Expected one of: Mamba, FLA_GLA"
        exit 1
        ;;
esac

if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then
    SAVE_DIR="$1"
    shift
else
    SAVE_DIR="${DEFAULT_SAVE_DIR}"
fi

cd "${TOOLS_DIR}"

python profile_lion.py \
    --cfg_file cfgs/carla_models/lion_lidar.yaml \
    --warmup 6 \
    --steps 6 \
    --save_dir "${SAVE_DIR}" \
    --set \
    MODEL.BACKBONE_3D.OPERATOR.NAME "${OPERATOR_NAME}" \
    MODEL.BACKBONE_3D.OPERATOR.USE_FUSED_MAPPING "${FUSED_MAPPING}" \
    "$@"
