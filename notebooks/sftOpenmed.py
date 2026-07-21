"""SFT OpenMed/privacy-filter-multilingual on our 10-type Chinese PII labels.

Warm-starts from OpenMed's multilingual checkpoint (zero-shot eval: ~97% region
detection + 0% strict bounding — it finds Chinese PII but shatters spans into
per-character / partial fragments under byte-BPE) and teaches it our full-span
Chinese bounding via supervised fine-tuning on our train.jsonl / val.jsonl.

Why not `opf train`: OpenMed publishes its checkpoint in HF transformers
format (config.json + safetensors + tokenizer files only — no opf-native
`original/` dir or `viterbi_calibration.json`), so `opf train --checkpoint`
can't ingest it. We use the standard HuggingFace Trainer path instead. Bonus:
this same harness is reusable as-is for the chinese-roberta rung — only the
`--base` flips.

Method: full fine-tuning. The classification head is reinitialized to OUR
10-type BIOES label space (41 labels: O + 10 × {B,I,E,S}); the encoder body
is warm-started from OpenMed. Hyperparameters mirror v1 so the comparison
isolates the starting checkpoint as the only variable.

Requires:
    pip install -U "openmed[hf]" "transformers>=5.9"  # 5.9 has native
                                                       # openai_privacy_filter

Run on Colab GPU (T4 or better). Save to mounted Drive for persistence:
    python notebooks/sftOpenmed.py \\
        --train data/sft/train.jsonl \\
        --val   data/sft/val.jsonl \\
        --output-dir /content/drive/MyDrive/piiclassifier_sft/finetuned_multilingual
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

OUR_TYPES = ["ADDRESS", "PHONE", "ID", "PERSON", "ORG",
             "DATE", "PLATE", "ACCOUNT", "EMAIL", "URL"]
# BIOES label space: O + 10 types × 4 prefixes = 41 labels
LABELS = ["O"] + [f"{p}-{t}" for t in OUR_TYPES for p in ("B", "I", "E", "S")]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

DEFAULT_BASE = "OpenMed/privacy-filter-multilingual"


def parse_spans(row: dict) -> list[tuple[int, int, str]]:
    """opf dict-span format -> [(start, end, type), ...].

    Keys are 'TYPE: surface text', values are list-of-intervals. Surface text
    is for human inspection only; we use the intervals.
    """
    out: list[tuple[int, int, str]] = []
    for k, intervals in (row.get("spans") or {}).items():
        t = k.split(":", 1)[0].strip()
        if t not in OUR_TYPES:
            continue
        for a, b in intervals:
            out.append((int(a), int(b), t))
    return out


def tokenize_and_tag(row: dict, tokenizer, max_length: int) -> dict:
    """Tokenize text + produce a BIOES label id per token.

    Tokens are tagged by which gold span their char-offsets fall in. A token
    that straddles a span boundary, or covers no characters (special tokens),
    is given -100 (O for cleanly-outside, -100 for special — both ignored in
    loss for special tokens).
    """
    text: str = row["text"]
    spans = parse_spans(row)

    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        max_length=max_length,
        truncation=True,
    )
    offsets: list[tuple[int, int]] = enc["offset_mapping"]

    # char index -> index into spans_sorted (or -1)
    n = len(text)
    spans_sorted = sorted(spans, key=lambda s: s[0])
    char_to_span = [-1] * n
    for si, (a, b, _t) in enumerate(spans_sorted):
        for j in range(max(0, a), min(n, b)):
            char_to_span[j] = si

    # token index -> index into spans_sorted (or -1 if outside/straddle/special)
    tok_spans: list[int] = []
    for (ts, te) in offsets:
        if ts == te:  # special tokens have empty offsets
            tok_spans.append(-1)
            continue
        chunk = char_to_span[ts:min(te, n)]
        if chunk and len(set(chunk)) == 1:
            tok_spans.append(chunk[0])
        else:  # straddles span boundary, or covers no in-span char
            tok_spans.append(-1)

    label_ids: list[int] = []
    for i, ((ts, te), sid) in enumerate(zip(offsets, tok_spans)):
        if ts == te:
            label_ids.append(-100)
            continue
        if sid == -1:
            label_ids.append(LABEL2ID["O"])
            continue
        is_first = (i == 0 or tok_spans[i - 1] != sid)
        is_last = (i == len(tok_spans) - 1 or tok_spans[i + 1] != sid)
        t = spans_sorted[sid][2]
        if is_first and is_last:
            lbl = f"S-{t}"
        elif is_first:
            lbl = f"B-{t}"
        elif is_last:
            lbl = f"E-{t}"
        else:
            lbl = f"I-{t}"
        label_ids.append(LABEL2ID[lbl])

    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": label_ids,
    }


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, items: list[dict]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        return self.items[i]


def load_jsonl(path: str, tokenizer, max_length: int) -> ListDataset:
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    items = [tokenize_and_tag(r, tokenizer, max_length) for r in rows]
    # Drop examples that tokenize to zero tokens (empty/whitespace text). This
    # tokenizer emits no special tokens for empty input, so such rows produce a
    # 0-length sequence; a batch made entirely of them crashes attention with
    # "cannot reshape tensor of 0 elements". Nothing to learn from them anyway.
    kept = [it for it in items if len(it["input_ids"]) > 0]
    dropped = len(items) - len(kept)
    if dropped:
        print(f"  dropped {dropped} empty-text rows from {path}")
    return ListDataset(kept)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--base", default=DEFAULT_BASE,
                    help="HF id or local dir of the base checkpoint to warm-start from")
    # HF full-FT of a from-scratch head needs a livelier schedule than v1's
    # opf run. v1's lr=1e-5 / no-warmup left the reinitialized head essentially
    # at init: grad_norm pinned ~200 vs max_grad_norm=1.0 throttled every step
    # to ~5e-8, so the head never learned the O-prior and predicted noise. A
    # higher lr + warmup + a few more epochs lets the head actually train.
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.1,
                    help="ramp lr from 0 to stabilize the huge initial grad_norm")
    ap.add_argument("--max-length", type=int, default=1024,
                    help="our complaint text is < 200 chars; 1024 is generous")
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"loading tokenizer + model from {args.base} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForTokenClassification.from_pretrained(
        args.base,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,  # reinitialize the classification head
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  body warm-started from {args.base}; "
          f"head reinitialized to {len(LABELS)} labels; "
          f"total params: {n_params / 1e9:.2f}B")

    print("\nloading + tagging training data ...")
    train_ds = load_jsonl(args.train, tokenizer, args.max_length)
    val_ds = load_jsonl(args.val, tokenizer, args.max_length)
    print(f"  train: {len(train_ds)} examples, val: {len(val_ds)}")

    collator = DataCollatorForTokenClassification(tokenizer, label_pad_token_id=-100)

    # bf16 only works on Ampere+ (A100/L4); T4 (Turing) needs fp16. Auto-detect.
    try:
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    except Exception:
        use_bf16 = False
    use_fp16 = torch.cuda.is_available() and not use_bf16
    print(f"\nprecision: bf16={use_bf16} fp16={use_fp16}")

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=args.epochs,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=tokenizer,  # transformers 5.x renamed `tokenizer=`
    )

    print("\nstarting training ...")
    trainer.train()

    print(f"\nsaving best model to {args.output_dir} ...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # finetune_summary.json for parity with v1's record
    summary = {
        "base_checkpoint": args.base,
        "method": "full fine-tuning via HF Trainer",
        "num_train_examples": len(train_ds),
        "num_val_examples": len(val_ds),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "learning_rate": args.lr,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "max_length": args.max_length,
        "span_class_names": ["O"] + OUR_TYPES,
        "num_output_labels": len(LABELS),
        "best_metric_name": "eval_loss",
        "best_metric": trainer.state.best_metric,
        "log_history": trainer.state.log_history,
    }
    with open(os.path.join(args.output_dir, "finetune_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("done.")


if __name__ == "__main__":
    main()
