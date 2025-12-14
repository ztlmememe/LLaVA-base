#!/usr/bin/env bash
# Stage 2: joint projector + LLM finetuning with optional LoRA on UniProtQA

set -euo pipefail

TRAIN_JSONL=${TRAIN_JSONL:-"data/uniprotqa_train.jsonl"}
SAVE_DIR=${SAVE_DIR:-"checkpoints/protein_finetune"}

python -m llava.train.protein_pipeline \
  --stage joint_finetune \
  --train_file "$TRAIN_JSONL" \
  --fm_model EvolutionaryScale/esmc-600m-2024-12 \
  --llm_model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --projector_type residual \
  --prefix_length 8 \
  --apply_lora \
  --lora_targets q_proj k_proj v_proj o_proj up_proj down_proj \
  --batch_size 1 \
  --projector_lr 1e-4 \
  --llm_lr 5e-6 \
  --epochs 1 \
  --save_dir "$SAVE_DIR"
