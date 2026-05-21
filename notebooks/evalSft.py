"""Evaluate a PII model on the held-out test set — strict span-F1.

Runs TWO models against the same gold, scored identically:
  1. baseline  — the un-fine-tuned openai/privacy-filter (native 8 English
                 labels, mapped to our schema). The "before fine-tuning" number.
  2. fine-tuned — your checkpoint (outputs our labels directly).

Strict span-F1: a prediction is a true positive only if (start, end, type)
all match a gold span exactly.

Two reporting scopes:
  in-schema   — only the 7 types the base model can produce (fair home-turf
                comparison: ADDRESS/PHONE/PERSON/EMAIL/URL/DATE/ACCOUNT)
  full-schema — all 10 of our types (base scores 0 recall on ID/PLATE/ORG by
                construction — it has no such categories)

Run on a GPU (Colab). opf must be installed; the base checkpoint auto-downloads.

    python notebooks/evalSft.py \
        --test /content/test_gold.jsonl \
        --checkpoint /content/drive/MyDrive/piiclassifier_sft/finetuned_checkpoint
"""
from __future__ import annotations

import argparse
import collections
import json
import os

OUR_TYPES = ["ADDRESS", "PHONE", "ID", "PERSON", "ORG",
             "DATE", "PLATE", "ACCOUNT", "EMAIL", "URL"]

# base model's native label -> our type. "secret" intentionally unmapped
# (we dropped SECRET); the base can't produce ID / PLATE / ORG at all.
OPF_TO_OURS = {
    "private_address": "ADDRESS", "private_phone": "PHONE",
    "private_person": "PERSON",   "private_email": "EMAIL",
    "private_url": "URL",         "private_date": "DATE",
    "account_number": "ACCOUNT",
}
IN_SCHEMA = set(OPF_TO_OURS.values())   # 7 types the base can produce
FULL_SCHEMA = set(OUR_TYPES)            # all 10 of ours


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return p, r, (2 * p * r / (p + r) if (p + r) else 0.0)


def score(gold_sets: list[set], pred_sets: list[set], allowed: set) -> dict:
    """Strict span-F1 over the `allowed` types. Spans of other types are
    dropped from both gold and pred before scoring."""
    tp = fp = fn = 0
    per = collections.defaultdict(lambda: [0, 0, 0])  # type -> [tp, fp, fn]
    for g_all, p_all in zip(gold_sets, pred_sets):
        g = {(a, b, t) for (a, b, t) in g_all if t in allowed}
        p = {(a, b, t) for (a, b, t) in p_all if t in allowed}
        tp += len(g & p); fp += len(p - g); fn += len(g - p)
        for (_, _, t) in (g & p): per[t][0] += 1
        for (_, _, t) in (p - g): per[t][1] += 1
        for (_, _, t) in (g - p): per[t][2] += 1
    f1s = [prf(*per[t])[2] for t in per if per[t][0] + per[t][2] > 0]
    return {
        "micro": prf(tp, fp, fn),
        "macro": sum(f1s) / len(f1s) if f1s else 0.0,
        "per": {t: prf(*per[t]) for t in per},
        "tot": (tp, fp, fn),
    }


def run_model(model, gold: list[dict], label_map: dict | None) -> list[set]:
    """Predict spans per row. label_map=None means the model already emits our
    type names (fine-tuned); a dict remaps the base model's native labels."""
    preds: list[set] = []
    for r in gold:
        s: set = set()
        if r["text"]:
            for sp in model.redact(r["text"]).detected_spans:
                t = label_map.get(sp.label) if label_map else sp.label
                if t in OUR_TYPES:
                    s.add((sp.start, sp.end, t))
        preds.append(s)
    return preds


def report(name: str, gold_sets: list[set], preds: list[set]) -> None:
    print(f"\n### {name} ###")
    for scope, allowed in [("in-schema  (7 base types)", IN_SCHEMA),
                           ("full-schema (all 10)", FULL_SCHEMA)]:
        res = score(gold_sets, preds, allowed)
        p, r, f = res["micro"]
        tp, fp, fn = res["tot"]
        print(f"  {scope:26} P={p:.3f} R={r:.3f} micro-F1={f:.3f} "
              f"macro-F1={res['macro']:.3f}  (TP={tp} FP={fp} FN={fn})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True, help="verified test_gold.jsonl")
    ap.add_argument("--checkpoint", required=True, help="fine-tuned checkpoint dir")
    ap.add_argument("--skip-baseline", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from opf import OPF

    gold = [json.loads(l) for l in open(args.test, encoding="utf-8")]
    gold_sets = [{(s["start"], s["end"], s["type"]) for s in r["spans"]} for r in gold]
    print(f"loaded {len(gold)} test rows, {sum(len(g) for g in gold_sets)} gold spans")

    results: dict[str, list[set]] = {}

    if not args.skip_baseline:
        print("\nrunning baseline (un-fine-tuned base model) ...")
        base = OPF(device="cuda", output_mode="typed", decode_mode="viterbi")
        results["BASELINE (un-fine-tuned)"] = run_model(base, gold, OPF_TO_OURS)

    print("running fine-tuned checkpoint ...")
    ft = OPF(model=args.checkpoint, device="cuda",
             output_mode="typed", decode_mode="viterbi")
    results["FINE-TUNED"] = run_model(ft, gold, None)

    print("\n" + "=" * 66)
    for name, preds in results.items():
        report(name, gold_sets, preds)

    # per-type for the fine-tuned model, full-schema
    print("\nfine-tuned per-type (full-schema):")
    res = score(gold_sets, results["FINE-TUNED"], FULL_SCHEMA)
    for t in OUR_TYPES:
        if t in res["per"]:
            p, r, f = res["per"][t]
            sup = res["per"][t]  # noqa
            print(f"  {t:>9}: P={p:.3f} R={r:.3f} F1={f:.3f}")
        else:
            print(f"  {t:>9}: (no gold spans in test)")


if __name__ == "__main__":
    main()
