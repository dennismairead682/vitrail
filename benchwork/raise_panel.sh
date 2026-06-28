#!/usr/bin/env bash
set -euo pipefail

NODES=${NODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CONFIG=${CONFIG:-configs/panels/main.yaml}

torchrun \
    --nnodes="${NODES}" \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR:-localhost}:${MASTER_PORT:-29501}" \
    -m vitrail.studio pretrain \
    --config "${CONFIG}" \
    --device cuda
