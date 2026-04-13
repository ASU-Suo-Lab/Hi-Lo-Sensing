#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ "$#" -lt 3 ]; then
    echo "Usage: bash scripts/carla_dsvt_eval.sh <gpus> <mha_legacy|sdpa|flash_attn> <ckpt> [extra_tag] [eval_tag] [additional test.py args...]"
    exit 1
fi

NGPUS="$1"
ATTN_BACKEND="$2"
CKPT="$3"
shift 3

case "${ATTN_BACKEND}" in
    mha_legacy)
        DEFAULT_EXTRA_TAG="carla_mha_legacy_train"
        DEFAULT_EVAL_TAG="carla_mha_legacy_eval"
        ;;
    sdpa)
        DEFAULT_EXTRA_TAG="carla_sdpa_train"
        DEFAULT_EVAL_TAG="carla_sdpa_eval"
        ;;
    flash_attn)
        DEFAULT_EXTRA_TAG="carla_flash_attn_fastest_train"
        DEFAULT_EVAL_TAG="carla_flash_attn_fastest_eval"
        ;;
    *)
        echo "Unsupported attention backend: ${ATTN_BACKEND}"
        echo "Expected one of: mha_legacy, sdpa, flash_attn"
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
    --cfg_file cfgs/carla_models/dsvt_pillar.yaml
    --ckpt "${CKPT}"
    --extra_tag "${EXTRA_TAG}"
    --eval_tag "${EVAL_TAG}"
    --set MODEL.BACKBONE_3D.ATTN_BACKEND "${ATTN_BACKEND}"
)

if [ "${NGPUS}" -gt 1 ]; then
    bash scripts/torch_test.sh "${NGPUS}" "${COMMON_ARGS[@]}" "$@"
else
    python test.py --launcher none "${COMMON_ARGS[@]}" "$@"
fi
