# Protein-to-Text Baseline

This folder contains a two-stage LLaVA-style pipeline that connects a protein foundation model (FM) to an LLM through a trainable projector.

## Dataset preparation
Each dataset has its own prep script and produces JSONL with a unified schema.

```json
{
  "id": "...",
  "protein_sequence": "...",
  "function_description": "...",
  "source_dataset": "..."
}
```

* `scripts/protein-baseline/data/prepare_mol_instructions.py --output data/mol_instructions_caption.jsonl`
* `scripts/protein-baseline/data/prepare_protdescribe.py --output data/protdescribe_caption.jsonl`
* `scripts/protein-baseline/data/prepare_prot2text.py --output data/prot2text_caption.jsonl`

UniProtQA keeps QA pairs:
```json
{
  "id": "...",
  "protein_sequence": "...",
  "question": "...",
  "answer": "...",
  "source_dataset": "UniProtQA"
}
```
* `scripts/protein-baseline/data/prepare_uniprotqa.py --output data/uniprotqa.jsonl`

## Prompts
Select a system prompt via `--system_prompt_path`:
* Caption: `scripts/protein-baseline/prompts/system/caption_default.txt`
* QA: `scripts/protein-baseline/prompts/system/qa_default.txt`

## Stage 1: projector pretraining
Freeze the LLM and learn the protein→LLM projector on a caption dataset.

```bash
bash scripts/protein-baseline/stage1_pretrain_projector.sh \
  DATA_PATH=data/mol_instructions_caption.jsonl \
  SYSTEM_PROMPT=scripts/protein-baseline/prompts/system/caption_default.txt \
  OUTPUT=checkpoints/protein_stage1
```

## Stage 2: finetuning
Train projector + LLM jointly (full finetuning or LoRA). Mix caption and QA JSONL files as needed.

Full finetune:
```bash
bash scripts/protein-baseline/stage2_finetune_full.sh \
  DATA_PATH=data/protein_mix_stage2.jsonl \
  SYSTEM_PROMPT=scripts/protein-baseline/prompts/system/qa_default.txt \
  OUTPUT=checkpoints/protein_stage2_full
```

LoRA finetune:
```bash
bash scripts/protein-baseline/stage2_finetune_lora.sh \
  DATA_PATH=data/protein_mix_stage2.jsonl \
  SYSTEM_PROMPT=scripts/protein-baseline/prompts/system/qa_default.txt \
  OUTPUT=checkpoints/protein_stage2_lora
```

### Debug mode
Add `DEBUG_PROMPT=True DEBUG_EVERY=200` to any training script to print formatted prompts and short generations during training.

## Evaluation
Prepare prediction JSONL files with `prediction` + reference fields.

Caption metrics (BLEU/ROUGE-L):
```bash
bash scripts/protein-baseline/eval_caption.sh PRED=predictions/caption_preds.jsonl
```

QA metrics (EM/F1):
```bash
bash scripts/protein-baseline/eval_qa.sh PRED=predictions/qa_preds.jsonl
```

## Model options
* Protein FM: `EvolutionaryScale/esmc-600m-2024-12` (default) or `facebook/esm2_t33_650M_UR50D` via `--protein_tower`.
* LLM: default `Llama-3.1-8B-Instruct`, optional `Qwen2.5-7B-Instruct`.
* Projector: `--mm_projector_type {linear,mlp,residual_mlp}` with `--mm_projector_prefix_length` and `--mm_projector_use_norm`.
