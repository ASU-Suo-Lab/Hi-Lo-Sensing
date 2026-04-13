#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ "$#" -lt 2 ]; then
    echo "Usage: bash scripts/carla_dsvt_train.sh <gpus> <mha_legacy|sdpa|flash_attn> [extra_tag] [additional train.py args...]"
    exit 1
fi

NGPUS="$1"
ATTN_BACKEND="$2"
shift 2

case "${ATTN_BACKEND}" in
    mha_legacy)
        DEFAULT_TAG_MULTI="carla_mha_legacy_train"
        DEFAULT_TAG_SINGLE="carla_mha_legacy_debug"
        ;;
    sdpa)
        DEFAULT_TAG_MULTI="carla_sdpa_train"
        DEFAULT_TAG_SINGLE="carla_sdpa_debug"
        ;;
    flash_attn)
        DEFAULT_TAG_MULTI="carla_flash_attn_fastest_train"
        DEFAULT_TAG_SINGLE="carla_flash_attn_fastest_debug"
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
    if [ "${NGPUS}" -gt 1 ]; then
        EXTRA_TAG="${DEFAULT_TAG_MULTI}"
    else
        EXTRA_TAG="${DEFAULT_TAG_SINGLE}"
    fi
fi

cd "${TOOLS_DIR}"

COMMON_ARGS=(
    --cfg_file cfgs/carla_models/dsvt_pillar.yaml
    --extra_tag "${EXTRA_TAG}"
    --set MODEL.BACKBONE_3D.ATTN_BACKEND "${ATTN_BACKEND}"
)

if [ "${NGPUS}" -gt 1 ]; then
    bash scripts/torch_train.sh "${NGPUS}" "${COMMON_ARGS[@]}" "$@"
else
    python train.py --launcher none "${COMMON_ARGS[@]}" "$@"
fi
