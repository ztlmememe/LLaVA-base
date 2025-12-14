#!/bin/bash
set -e

DATA_PATH=${DATA_PATH:-"data/mol_instructions_caption.jsonl"}
SYSTEM_PROMPT=${SYSTEM_PROMPT:-"scripts/protein-baseline/prompts/system/caption_default.txt"}
OUTPUT=${OUTPUT:-"checkpoints/protein_stage1"}

python -m llava.train.train_mem \
    --model_name_or_path "meta-llama/Llama-3.1-8B-Instruct" \
    --protein_tower "EvolutionaryScale/esmc-600m-2024-12" \
    --mm_projector_type "mlp2x_gelu" \
    --mm_projector_prefix_length 8 \
    --mm_projector_use_norm True \
    --data_path ${DATA_PATH} \
    --protein_baseline True \
    --system_prompt_path ${SYSTEM_PROMPT} \
    --task_template caption \
    --bf16 True \
    --output_dir ${OUTPUT} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --mm_projector_lr 2e-4 \
    --freeze_backbone True \
    --logging_steps 10 \
    --save_steps 500 \
    --model_max_length 2048 \
    --debug_prompt ${DEBUG_PROMPT:-False} \
    --debug_every_n_steps ${DEBUG_EVERY:-200}
