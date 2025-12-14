import argparse
import json
from collections import Counter
import re


def normalize(text):
    return re.sub(r"\s+", " ", text.strip().lower())


def f1_score(prediction, ground_truth):
    pred_tokens = normalize(prediction).split()
    truth_tokens = normalize(ground_truth).split()
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def exact_match(prediction, ground_truth):
    return normalize(prediction) == normalize(ground_truth)


def main():
    parser = argparse.ArgumentParser(description="Evaluate QA predictions with EM and F1")
    parser.add_argument("--predictions", required=True, help="JSONL file with fields prediction and answer")
    args = parser.parse_args()

    em_total = 0
    f1_total = 0.0
    count = 0
    with open(args.predictions, "r") as f:
        for line in f:
            row = json.loads(line)
            answer = row.get("answer")
            pred = row.get("prediction")
            if answer is None or pred is None:
                continue
            em_total += int(exact_match(pred, answer))
            f1_total += f1_score(pred, answer)
            count += 1
    metrics = {
        "exact_match": em_total / count if count else 0.0,
        "f1": f1_total / count if count else 0.0
    }
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
