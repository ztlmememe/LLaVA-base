#!/bin/bash
set -e

PRED=${PRED:-"predictions/qa_preds.jsonl"}
python scripts/protein-baseline/eval_qa.py --predictions ${PRED}
