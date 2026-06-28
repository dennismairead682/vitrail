#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-configs/panels/main.yaml}
TASK=${TASK:-seg_cbct}
PANE=${PANE:-runs/main_encoder.pane}

python -m vitrail.studio finetune \
    --config "${CONFIG}" \
    --task "${TASK}" \
    --pretrained "${PANE}" \
    --device cuda

python -m vitrail.studio appraise \
    --config "${CONFIG}" \
    --task "${TASK}" \
    --pane "runs/$(basename "${CONFIG%.yaml}")_${TASK}.pane" \
    --device cuda
