import argparse
import json
from collections import Counter
import math


def ngram_counts(tokens, n):
    return Counter([tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)])


def corpus_bleu(references, predictions, max_n=4):
    weights = [1.0 / max_n] * max_n
    p_ns = []
    for n in range(1, max_n + 1):
        match = 0
        total = 0
        for ref, pred in zip(references, predictions):
            ref_counts = ngram_counts(ref, n)
            pred_counts = ngram_counts(pred, n)
            overlap = pred_counts & ref_counts
            match += sum(overlap.values())
            total += max(1, sum(pred_counts.values()))
        p_ns.append(match / total if total > 0 else 0.0)
    geo_mean = sum(w * math.log(p + 1e-12) for w, p in zip(weights, p_ns))
    bp = 1.0
    ref_len = sum(len(r) for r in references)
    pred_len = sum(len(p) for p in predictions)
    if pred_len < ref_len:
        bp = math.exp(1 - ref_len / max(pred_len, 1))
    return bp * math.exp(geo_mean)


def rouge_l_score(reference, prediction):
    ref_len, pred_len = len(reference), len(prediction)
    lcs_grid = [[0] * (pred_len + 1) for _ in range(ref_len + 1)]
    for i in range(ref_len):
        for j in range(pred_len):
            if reference[i] == prediction[j]:
                lcs_grid[i + 1][j + 1] = lcs_grid[i][j] + 1
            else:
                lcs_grid[i + 1][j + 1] = max(lcs_grid[i][j + 1], lcs_grid[i + 1][j])
    lcs = lcs_grid[-1][-1]
    prec = lcs / max(pred_len, 1)
    rec = lcs / max(ref_len, 1)
    if prec + rec == 0:
        return 0.0
    return (2 * prec * rec) / (prec + rec)


def main():
    parser = argparse.ArgumentParser(description="Evaluate caption predictions with BLEU and ROUGE-L")
    parser.add_argument("--predictions", required=True, help="JSONL file with fields prediction and function_description")
    args = parser.parse_args()

    references = []
    predictions = []
    with open(args.predictions, "r") as f:
        for line in f:
            row = json.loads(line)
            ref = row.get("function_description") or row.get("reference")
            pred = row.get("prediction")
            if ref is None or pred is None:
                continue
            references.append(ref.strip().split())
            predictions.append(pred.strip().split())

    bleu = corpus_bleu(references, predictions) if references else 0.0
    rouge_scores = [rouge_l_score(r, p) for r, p in zip(references, predictions)]
    rouge_l = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0

    print(json.dumps({"bleu": bleu, "rouge_l": rouge_l}, indent=2))


if __name__ == "__main__":
    main()
