#!/usr/bin/env bash
# Stage 1: projector-only training on Swiss-Prot captions

set -euo pipefail

TRAIN_JSONL=${TRAIN_JSONL:-"data/swissprot_caption.jsonl"}
SAVE_DIR=${SAVE_DIR:-"checkpoints/protein_pretrain"}

python -m llava.train.protein_pipeline \
  --stage projector_pretrain \
  --train_file "$TRAIN_JSONL" \
  --fm_model EvolutionaryScale/esmc-600m-2024-12 \
  --llm_model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --projector_type mlp2x_gelu \
  --prefix_length 8 \
  --freeze_fm \
  --batch_size 2 \
  --projector_lr 2e-4 \
  --epochs 1 \
  --save_dir "$SAVE_DIR"
