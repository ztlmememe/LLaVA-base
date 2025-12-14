import argparse
import json
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Prepare Prot2Text into unified protein caption JSONL")
    parser.add_argument("--output", required=True, help="Path to write JSONL output")
    args = parser.parse_args()

    dataset = load_dataset("habdine/Prot2Text-Data", split="train")
    with open(args.output, "w") as f:
        for idx, row in enumerate(dataset):
            seq = row.get("protein_sequence") or row.get("sequence")
            desc = row.get("function_description") or row.get("text")
            if not seq or not desc:
                continue
            seq = seq.strip()
            desc = desc.strip()
            if not seq or not desc:
                continue
            example = {
                "id": row.get("id", f"prot2text_{idx}"),
                "protein_sequence": seq,
                "function_description": desc,
                "source_dataset": "Prot2Text"
            }
            f.write(json.dumps(example) + "\n")


if __name__ == "__main__":
    main()
