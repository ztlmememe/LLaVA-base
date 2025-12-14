#!/bin/bash
set -e

DATA_PATH=${DATA_PATH:-"data/protein_mix_stage2.jsonl"}
SYSTEM_PROMPT=${SYSTEM_PROMPT:-"scripts/protein-baseline/prompts/system/qa_default.txt"}
OUTPUT=${OUTPUT:-"checkpoints/protein_stage2_full"}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-"./scripts/zero3.json"}
REPORT_TO=${REPORT_TO:-"wandb"}

python -m llava.train.train_mem \
    --model_name_or_path "meta-llama/Llama-3.1-8B-Instruct" \
    --protein_tower "EvolutionaryScale/esmc-600m-2024-12" \
    --mm_projector_type "residual_mlp" \
    --mm_projector_prefix_length 8 \
    --mm_projector_use_norm True \
    --data_path ${DATA_PATH} \
    --protein_baseline True \
    --system_prompt_path ${SYSTEM_PROMPT} \
    --task_template qa \
    --version v1 \
    --bf16 True \
    --output_dir ${OUTPUT} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --mm_projector_lr 1e-4 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --group_by_modality_length True \
    --tf32 True \
    --report_to ${REPORT_TO} \
    --deepspeed ${DEEPSPEED_CONFIG} \
    --debug_prompt ${DEBUG_PROMPT:-False} \
    --debug_every_n_steps ${DEBUG_EVERY:-200}
