import argparse
import json
from pathlib import Path
from typing import List

import torch

from llava.train.protein_pipeline import ProteinToTextModel, build_examples, read_jsonl


def f1_score(pred: str, truth: str) -> float:
    pred_tokens = pred.split()
    truth_tokens = truth.split()
    common = set(pred_tokens) & set(truth_tokens)
    if len(common) == 0:
        return 0.0
    prec = len(common) / len(pred_tokens)
    rec = len(common) / len(truth_tokens)
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
    examples = build_examples(raw, stage="joint_finetune")

    f1_scores, exact_matches = [], []
    for ex in examples:
        with torch.no_grad():
            fm_hidden = model.fm.embed([ex.sequence], device=device)
            prefix = model.projector(fm_hidden).to(device)
            input_ids = model.llm_tokenizer(ex.prompt, return_tensors="pt").input_ids.to(device)
            prompt_embeds = model.llm.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat([prefix, prompt_embeds], dim=1)
            attn = torch.ones(inputs_embeds.shape[:2], device=device)
            generation = model.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attn,
                max_new_tokens=args.max_new_tokens,
            )
            decoded = model.llm_tokenizer.decode(generation[0], skip_special_tokens=True)
        f1_scores.append(f1_score(decoded, ex.target))
        exact_matches.append(float(decoded.strip().lower() == ex.target.strip().lower()))

    print(json.dumps({
        "f1": sum(f1_scores) / len(f1_scores),
        "exact_match": sum(exact_matches) / len(exact_matches),
    }, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate UniProtQA-style QA")
    parser.add_argument("--data", type=str, required=True, help="JSONL with sequence/question/answer")
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
