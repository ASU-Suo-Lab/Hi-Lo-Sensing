#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ "$#" -lt 3 ]; then
    echo "Usage: bash scripts/carla_lion_eval.sh <gpus> <Mamba|FLA_GLA> <ckpt> [extra_tag] [eval_tag] [additional test.py args...]"
    exit 1
fi

NGPUS="$1"
OPERATOR_NAME="$2"
CKPT="$3"
shift 3

case "${OPERATOR_NAME}" in
    Mamba)
        DEFAULT_EXTRA_TAG="carla_lion_mamba_train"
        DEFAULT_EVAL_TAG="carla_lion_mamba_eval"
        FUSED_MAPPING="False"
        ;;
    FLA_GLA)
        DEFAULT_EXTRA_TAG="carla_lion_fla_train"
        DEFAULT_EVAL_TAG="carla_lion_fla_eval"
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
    EXTRA_TAG="${DEFAULT_EXTRA_TAG}"
fi

if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then
    EVAL_TAG="$1"
    shift
else
    EVAL_TAG="${DEFAULT_EVAL_TAG}"
fi

cd "${TOOLS_DIR}"

COMMON_ARGS=(
    --cfg_file cfgs/carla_models/lion_lidar.yaml
    --ckpt "${CKPT}"
    --extra_tag "${EXTRA_TAG}"
    --eval_tag "${EVAL_TAG}"
    --set
    MODEL.BACKBONE_3D.OPERATOR.NAME "${OPERATOR_NAME}"
    MODEL.BACKBONE_3D.OPERATOR.USE_FUSED_MAPPING "${FUSED_MAPPING}"
)

if [ "${NGPUS}" -gt 1 ]; then
    bash scripts/torch_test.sh "${NGPUS}" "${COMMON_ARGS[@]}" "$@"
else
    python test.py --launcher none "${COMMON_ARGS[@]}" "$@"
fi
