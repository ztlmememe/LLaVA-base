# Protein FM \+ LLM baseline

This document describes how to train and evaluate a protein-to-text system that mirrors the two-stage LLaVA pipeline while replacing the visual tower with a protein foundation model (FM).

## Components

- **Protein encoder (FM)**: `EvolutionaryScale/esmc-600m-2024-12` (default) or `facebook/esm2_t33_650M_UR50D`.
- **LLM decoder**: `meta-llama/Meta-Llama-3.1-8B-Instruct` (default) or `Qwen2.5-7B-Instruct` via the `--llm_model` flag.
- **Projector**: configurable (`linear`, `mlp2x_gelu`, `residual`) mapping pooled FM embeddings to LLM prefix tokens.
- **LoRA (optional)**: attachable to attention/FFN blocks during joint finetuning.

## Data format

Inputs are JSONL files with per-line dictionaries:

- **Swiss-Prot captioning** (stage 1):

```json
{"sequence": "MSEQN...", "caption": "cytoplasmic kinase that binds ATP"}
```

- **UniProtQA** (stage 2):

```json
{"sequence": "MSEQN...", "question": "What is the subcellular location?", "answer": "Periplasm"}
```

## Training

### Stage 1 – projector pretraining (LLM frozen)

```bash
bash scripts/protein_pretrain.sh
# Environment overrides: TRAIN_JSONL, SAVE_DIR
```

Key flags:
- `--stage projector_pretrain` keeps the LLM fixed and only optimizes the projector (and FM if `--freeze_fm` is omitted).
- `--projector_type` selects the mapping depth; `--prefix_length` controls how many prefix tokens are fused before the caption text.

### Stage 2 – joint finetuning (optional LoRA)

```bash
bash scripts/protein_finetune.sh
# Environment overrides: TRAIN_JSONL, SAVE_DIR
```

Key flags:
- `--stage joint_finetune` updates the projector and LLM parameters.
- `--apply_lora` with `--lora_targets` attaches adapters to the listed modules (default covers attention and FFN blocks).
- Adjust `--llm_lr` and `--projector_lr` independently.

## Evaluation

### Captioning (Swiss-Prot)

```bash
python scripts/eval_protein_caption.py \
  --data data/swissprot_caption_val.jsonl \
  --checkpoint checkpoints/protein_pretrain/checkpoint_epoch0.pt \
  --prefix_length 8
```

Outputs BLEU-1 and ROUGE-L computed on decoded captions.

### QA (UniProtQA)

```bash
python scripts/eval_protein_qa.py \
  --data data/uniprotqa_val.jsonl \
  --checkpoint checkpoints/protein_finetune/checkpoint_epoch0.pt \
  --prefix_length 8
```

Reports token-level F1 and exact match.

## Configuration highlights

- Swap encoders/decoders by changing `--fm_model` or `--llm_model`.
- Control sequence truncation via `--fm_max_length`.
- Prefix token behavior is governed by `--prefix_length`, and projector architecture by `--projector_type`.
- Use `--freeze_fm` in stage 2 to keep the FM fixed when only projector/LLM should update.
