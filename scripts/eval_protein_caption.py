import argparse
import json
from collections import Counter
from pathlib import Path
from typing import List

import torch

from llava.train.protein_pipeline import ProteinToTextModel, build_examples, read_jsonl


def bleu1(candidate: List[str], reference: List[str]) -> float:
    cand_counts = Counter(candidate)
    ref_counts = Counter(reference)
    overlap = sum((cand_counts & ref_counts).values())
    return overlap / max(len(candidate), 1)


def rouge_l(candidate: List[str], reference: List[str]) -> float:
    dp = [[0] * (len(reference) + 1) for _ in range(len(candidate) + 1)]
    for i, c in enumerate(candidate, 1):
        for j, r in enumerate(reference, 1):
            if c == r:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[-1][-1]
    prec = lcs / max(len(candidate), 1)
    rec = lcs / max(len(reference), 1)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def evaluate(args):
    device = torch.device(args.device)
    model = ProteinToTextModel(
        fm_model=args.fm_model,
        llm_model=args.llm_model,
        projector_type=args.projector_type,
        prefix_length=args.prefix_length,
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    raw = read_jsonl(Path(args.data))
    examples = build_examples(raw, stage="projector_pretrain")

    bleu_scores, rouge_scores = [], []
    for ex in examples:
        with torch.no_grad():
            fm_hidden = model.fm.embed([ex.sequence], device=device)
            prefix = model.projector(fm_hidden).to(device)

            prompt = ex.prompt
            input_ids = model.llm_tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            prompt_embeds = model.llm.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat([prefix, prompt_embeds], dim=1)
            attn = torch.ones(inputs_embeds.shape[:2], device=device)
            generation = model.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attn,
                max_new_tokens=args.max_new_tokens,
            )
            decoded = model.llm_tokenizer.decode(generation[0], skip_special_tokens=True)
        pred_tokens = decoded.split()
        ref_tokens = ex.target.split()
        bleu_scores.append(bleu1(pred_tokens, ref_tokens))
        rouge_scores.append(rouge_l(pred_tokens, ref_tokens))

    print(json.dumps({
        "bleu1": sum(bleu_scores) / len(bleu_scores),
        "rouge_l": sum(rouge_scores) / len(rouge_scores),
    }, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate protein captioning")
    parser.add_argument("--data", type=str, required=True, help="JSONL with sequence/caption")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--fm_model", type=str, default="EvolutionaryScale/esmc-600m-2024-12")
    parser.add_argument("--llm_model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--projector_type", type=str, default="linear")
    parser.add_argument("--prefix_length", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
