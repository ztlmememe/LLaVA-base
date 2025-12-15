import argparse
import json
from datasets import load_dataset


def filter_record(record):
    seq = record.get("protein_sequence") or record.get("sequence") or record.get("target_sequence")
    desc = record.get("function_description") or record.get("instruction") or record.get("description")
    if not seq or not desc:
        return False
    seq = seq.strip()
    desc = desc.strip()
    return len(seq) > 0 and len(desc) > 0


def main():
    parser = argparse.ArgumentParser(description="Prepare Mol-Instructions into unified protein caption JSONL")
    parser.add_argument("--output", required=True, help="Path to write JSONL output")
    args = parser.parse_args()

    dataset = load_dataset("zjunlp/Mol-Instructions", split="train")
    with open(args.output, "w") as f:
        for idx, row in enumerate(dataset):
            seq = row.get("protein_sequence") or row.get("sequence") or row.get("target_sequence")
            desc = row.get("function_description") or row.get("instruction") or row.get("description")
            if not filter_record({"protein_sequence": seq, "function_description": desc}):
                continue
            example = {
                "id": row.get("id", f"mol_inst_{idx}"),
                "protein_sequence": seq.strip(),
                "function_description": desc.strip(),
                "source_dataset": "Mol-Instructions"
            }
            f.write(json.dumps(example) + "\n")

if __name__ == "__main__":
    main()
