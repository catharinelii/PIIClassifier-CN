"""Strict span-F1 eval for a plain HuggingFace token-classification checkpoint.

This is the correct evaluator for models trained by `sftOpenmed.py` — i.e. a
standard `AutoModelForTokenClassification` whose head we reinitialized to our
41-label BIOES space (O + 10 types x {B,I,E,S}).

Why not evalSft / evalOpenmed:
  - evalSft.py decodes through the `opf` library (opf-native checkpoints only).
  - evalOpenmed.py decodes through OpenMed's PrivacyFilterTorchPipeline, whose
    BIOES-Viterbi grouping is calibrated to OpenMed's OWN 217-label scheme. Fed
    our reinitialized 41-label head, it can't decode and emits nothing — the
    all-zeros (TP=0 AND FP=0) symptom.
Neither runs a vanilla HF token-classifier, which is exactly what we trained.

Decoding: tokenize with char offsets, argmax per token -> BIOES tag -> merge
into (start, end, type) char spans, mirroring the tagging in sftOpenmed.py in
reverse. Scored with the IDENTICAL strict span-F1 imported from evalSft, so the
number lines up directly against BASELINE, v1, and zero-shot OpenMed.

    pip install -U "transformers>=5.9"
    python notebooks/evalHf.py \
        --test data/sft/test_gold.jsonl \
        --model /content/drive/MyDrive/piiclassifier_sft/finetuned_multilingual
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

# Reuse the IDENTICAL scoring code so this is apples-to-apples with every other
# eval. (evalSft imports `opf` only inside main(), so importing the module here
# is cheap and does NOT require opf to be installed.)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evalSft import score, report, OUR_TYPES, FULL_SCHEMA  # noqa: E402


def decode_bioes(offsets, label_ids, id2label):
    """Per-token BIOES tags -> list of (start, end, type) char spans.

    Lenient on malformed tag sequences (argmax can emit I/E without a B): a
    same-type continuation extends the open span; a type change or an O closes
    it. S and B always start a fresh span so two adjacent same-type entities
    don't merge.
    """
    spans: list[tuple[int, int, str]] = []
    cur: list | None = None  # [type, start, end]

    def close():
        nonlocal cur
        if cur is not None:
            spans.append((cur[1], cur[2], cur[0]))
            cur = None

    for (ts, te), lid in zip(offsets, label_ids):
        if ts == te:  # special / zero-width token
            continue
        lab = id2label[lid]
        if lab == "O":
            close()
            continue
        prefix, _, typ = lab.partition("-")
        if typ not in OUR_TYPES:
            close()
            continue
        if prefix == "S":
            close()
            spans.append((ts, te, typ))
        elif prefix == "B":
            close()
            cur = [typ, ts, te]
        else:  # I or E
            if cur is not None and cur[0] == typ:
                cur[2] = te
            else:
                close()
                cur = [typ, ts, te]
            if prefix == "E":
                close()
    close()
    return spans


def run_hf(model, tokenizer, gold, device, max_length, show=0):
    id2label = model.config.id2label
    preds: list[set] = []
    shown = 0
    for r in gold:
        s: set = set()
        if r["text"]:
            enc = tokenizer(
                r["text"],
                return_offsets_mapping=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            offsets = enc.pop("offset_mapping")[0].tolist()
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                logits = model(**enc).logits[0]
            label_ids = logits.argmax(-1).tolist()
            for (a, b, t) in decode_bioes(offsets, label_ids, id2label):
                if b > a:
                    s.add((a, b, t))
            if show and shown < show and s:
                print(f"\nrow {r.get('id', '?')}: {r['text'][:80]!r}")
                for (a, b, t) in sorted(s):
                    print(f"    {t:>8} [{a}:{b}]  {r['text'][a:b]!r}")
                shown += 1
        preds.append(s)
    return preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True, help="verified test_gold.jsonl")
    ap.add_argument("--model", required=True, help="HF token-classification checkpoint dir")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--show", type=int, default=3, help="print predictions for the first N non-empty rows")
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    gold = [json.loads(l) for l in open(args.test, encoding="utf-8")]
    gold_sets = [{(s["start"], s["end"], s["type"]) for s in r["spans"]} for r in gold]
    print(f"loaded {len(gold)} test rows, {sum(len(g) for g in gold_sets)} gold spans")

    print(f"\nloading {args.model} on {args.device} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(args.model).to(args.device).eval()
    print(f"  {model.config.num_labels} labels; e.g. "
          f"{[model.config.id2label[i] for i in range(min(5, model.config.num_labels))]}")

    preds = run_hf(model, tokenizer, gold, args.device, args.max_length, args.show)

    print("\n" + "=" * 66)
    report(args.model, gold_sets, preds)

    print(f"\n{args.model} per-type (full-schema):")
    res = score(gold_sets, preds, FULL_SCHEMA)
    for t in OUR_TYPES:
        if t in res["per"]:
            p, r, f = res["per"][t]
            print(f"  {t:>9}: P={p:.3f} R={r:.3f} F1={f:.3f}")
        else:
            print(f"  {t:>9}: (no gold spans in test)")


if __name__ == "__main__":
    main()
