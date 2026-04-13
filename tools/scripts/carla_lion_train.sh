#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ "$#" -lt 2 ]; then
    echo "Usage: bash scripts/carla_lion_train.sh <gpus> <Mamba|FLA_GLA> [extra_tag] [additional train.py args...]"
    exit 1
fi

NGPUS="$1"
OPERATOR_NAME="$2"
shift 2

case "${OPERATOR_NAME}" in
    Mamba)
        DEFAULT_TAG_MULTI="carla_lion_mamba_train"
        DEFAULT_TAG_SINGLE="carla_lion_mamba_debug"
        FUSED_MAPPING="False"
        ;;
    FLA_GLA)
        DEFAULT_TAG_MULTI="carla_lion_fla_train"
        DEFAULT_TAG_SINGLE="carla_lion_fla_debug"
        FUSED_MAPPING="True"
        ;;
    *)
        echo "Unsupported LION operator: ${OPERATOR_NAME}"
        echo "Expected one of: Mamba, FLA_GLA"
        exit 1
        ;;
esac

if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then
    EXTRA_TAG="$1"
    shift
else
    if [ "${NGPUS}" -gt 1 ]; then
        EXTRA_TAG="${DEFAULT_TAG_MULTI}"
    else
        EXTRA_TAG="${DEFAULT_TAG_SINGLE}"
    fi
fi

cd "${TOOLS_DIR}"

COMMON_ARGS=(
    --cfg_file cfgs/carla_models/lion_lidar.yaml
    --extra_tag "${EXTRA_TAG}"
    --set
    MODEL.BACKBONE_3D.OPERATOR.NAME "${OPERATOR_NAME}"
    MODEL.BACKBONE_3D.OPERATOR.USE_FUSED_MAPPING "${FUSED_MAPPING}"
)

if [ "${NGPUS}" -gt 1 ]; then
    bash scripts/torch_train.sh "${NGPUS}" "${COMMON_ARGS[@]}" "$@"
else
    python train.py --launcher none "${COMMON_ARGS[@]}" "$@"
fi
