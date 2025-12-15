#!/bin/bash
set -e

PRED=${PRED:-"predictions/caption_preds.jsonl"}
python scripts/protein-baseline/eval_caption.py --predictions ${PRED}
