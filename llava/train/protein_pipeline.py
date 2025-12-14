import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from llava.model.protein_encoder.builder import build_protein_encoder
from llava.model.protein_projector import ProteinProjector


@dataclass
class TrainingExample:
    sequence: str
    prompt: str
    target: str


class ProteinTextDataset(Dataset):
    def __init__(self, examples: List[TrainingExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> TrainingExample:
        return self.examples[idx]


class ProteinToTextModel(nn.Module):
    def __init__(
        self,
        fm_model: str,
        llm_model: str,
        projector_type: str = "linear",
        prefix_length: int = 8,
        fm_max_length: int = 4096,
        apply_lora: bool = False,
        lora_targets: Optional[List[str]] = None,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.fm = build_protein_encoder(fm_model, max_length=fm_max_length)
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model)
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model)
        hidden_size = self.llm.config.hidden_size
        self.projector = ProteinProjector(
            input_dim=self.fm.hidden_size,
            output_dim=hidden_size,
            prefix_length=prefix_length,
            projector_type=projector_type,
        )

        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
            self.llm.config.pad_token_id = self.llm_tokenizer.pad_token_id

        if apply_lora:
            lora_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_targets or ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
            )
            self.llm = get_peft_model(self.llm, lora_cfg)

    def freeze_llm(self):
        for p in self.llm.parameters():
            p.requires_grad = False

    def freeze_fm(self):
        for p in self.fm.parameters():
            p.requires_grad = False

    def forward(
        self,
        sequences: List[str],
        prompts: List[str],
        targets: List[str],
        device: torch.device,
    ) -> torch.Tensor:
        fm_hidden = self.fm(sequences, device=device, require_grad=any(p.requires_grad for p in self.fm.parameters()))
        prefix = self.projector(fm_hidden)

        text_inputs = [prompt + target for prompt, target in zip(prompts, targets)]
        encoded = self.llm_tokenizer(
            text_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.llm.config.max_position_embeddings,
        )
        input_ids = encoded.input_ids.to(device)
        attention_mask = encoded.attention_mask.to(device)

        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        batch_size = inputs_embeds.size(0)
        prefix = prefix.to(device)
        expanded_attention = torch.ones(batch_size, prefix.size(1), device=device, dtype=attention_mask.dtype)

        inputs_embeds = torch.cat([prefix, inputs_embeds], dim=1)
        attention_mask = torch.cat([expanded_attention, attention_mask], dim=1)

        labels = input_ids.clone()
        labels[input_ids == self.llm_tokenizer.pad_token_id] = -100
        prefix_labels = torch.full((batch_size, prefix.size(1)), -100, device=device, dtype=labels.dtype)
        labels = torch.cat([prefix_labels, labels], dim=1)

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        return outputs.loss


def read_jsonl(path: Path) -> List[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def build_examples(raw: Iterable[dict], stage: str) -> List[TrainingExample]:
    examples: List[TrainingExample] = []
    if stage == "projector_pretrain":
        for row in raw:
            examples.append(
                TrainingExample(
                    sequence=row["sequence"],
                    prompt="",  # caption generation does not need a question prefix
                    target=row["caption"],
                )
            )
    else:
        for row in raw:
            question = row.get("question") or ""
            prompt = f"Question: {question}\nAnswer: "
            examples.append(
                TrainingExample(
                    sequence=row["sequence"],
                    prompt=prompt,
                    target=row["answer"],
                )
            )
    return examples


def collate_fn(examples: List[TrainingExample]):
    sequences = [ex.sequence for ex in examples]
    prompts = [ex.prompt for ex in examples]
    targets = [ex.target for ex in examples]
    return sequences, prompts, targets


def train(args):
    set_seed(args.seed)
    device = torch.device(args.device)
    raw_examples = read_jsonl(Path(args.train_file))
    examples = build_examples(raw_examples, args.stage)
    dataset = ProteinTextDataset(examples)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    model = ProteinToTextModel(
        fm_model=args.fm_model,
        llm_model=args.llm_model,
        projector_type=args.projector_type,
        prefix_length=args.prefix_length,
        fm_max_length=args.fm_max_length,
        apply_lora=args.apply_lora,
        lora_targets=args.lora_targets,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    ).to(device)

    if args.stage == "projector_pretrain":
        model.freeze_llm()
    if args.freeze_fm:
        model.freeze_fm()

    optim_groups = [
        {"params": model.projector.parameters(), "lr": args.projector_lr},
    ]
    if args.stage == "joint_finetune":
        optim_groups.append({"params": [p for p in model.llm.parameters() if p.requires_grad], "lr": args.llm_lr})
    optimizer = torch.optim.AdamW(optim_groups, weight_decay=args.weight_decay)

    model.train()
    global_step = 0
    for epoch in range(args.epochs):
        for batch in dataloader:
            sequences, prompts, targets = batch
            loss = model(sequences, prompts, targets, device=device)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            if global_step % args.log_every == 0:
                print(f"Epoch {epoch} step {global_step} loss {loss.item():.4f}")

        if args.save_dir:
            save_path = Path(args.save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path / f"checkpoint_epoch{epoch}.pt")
            model.llm.save_pretrained(save_path / "llm")
            model.llm_tokenizer.save_pretrained(save_path / "llm")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Protein-to-Text training pipeline")
    parser.add_argument("--stage", choices=["projector_pretrain", "joint_finetune"], required=True)
    parser.add_argument("--train_file", type=str, required=True, help="JSONL file with sequence/question/answer fields")
    parser.add_argument("--fm_model", type=str, default="EvolutionaryScale/esmc-600m-2024-12")
    parser.add_argument("--llm_model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--projector_type", type=str, default="linear", choices=["linear", "residual", "mlp2x_gelu"])
    parser.add_argument("--prefix_length", type=int, default=8)
    parser.add_argument("--fm_max_length", type=int, default=4096)
    parser.add_argument("--apply_lora", action="store_true")
    parser.add_argument("--lora_targets", nargs="*", default=None)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--freeze_fm", action="store_true")
    parser.add_argument("--projector_lr", type=float, default=2e-4)
    parser.add_argument("--llm_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="")
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
