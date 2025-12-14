import argparse
import json
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Prepare UniProtQA JSONL")
    parser.add_argument("--output", required=True, help="Path to write JSONL output")
    args = parser.parse_args()

    dataset = load_dataset("PharMolix/UniProtQA", split="train")
    with open(args.output, "w") as f:
        for idx, row in enumerate(dataset):
            seq = row.get("protein_sequence") or row.get("sequence")
            question = row.get("question") or row.get("instruction")
            answer = row.get("answer") or row.get("response")
            if not seq or not question or not answer:
                continue
            seq = seq.strip()
            question = question.strip()
            answer = answer.strip()
            if not seq or not question or not answer:
                continue
            example = {
                "id": row.get("id", f"uniprotqa_{idx}"),
                "protein_sequence": seq,
                "question": question,
                "answer": answer,
                "source_dataset": "UniProtQA"
            }
            f.write(json.dumps(example) + "\n")


if __name__ == "__main__":
    main()
