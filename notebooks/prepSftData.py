"""Prepare the SFT dataset for the OpenAI privacy-filter's `opf train`.

Combines our labeled data into `opf`'s expected schema and carves out a
held-out test set.

Inputs:
  data/gold.jsonl          298 rows (Codex-labeled + ORG-cleaned)
  data/llm_labeled.jsonl   ~2000 rows (gpt-5.4-mini-labeled + post-filtered)

Outputs (data/sft/):
  train.jsonl       opf format — the bulk; what the model learns from
  val.jsonl         opf format — held out, watched during training
  label_space.json  our 11-type label space (for --label-space-json)
  test_gold.jsonl   the 50 held-out test rows, in OUR schema. These need
                    HUMAN VERIFICATION before they can evaluate anything —
                    so they are NOT converted to opf format here.

The 50 test rows are drawn from gold.jsonl (stratified by length) and
excluded from train/val. opf's span schema is a dict keyed by "LABEL: surface"
mapping to a list of [start, end] character-offset pairs.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "gold.jsonl"
LLM = ROOT / "data" / "llm_labeled.jsonl"
OUT_DIR = ROOT / "data" / "sft"

SEED = 42
N_TEST = 50
VAL_FRACTION = 0.10

OUR_TYPES = ["ADDRESS", "PHONE", "ID", "PERSON", "ORG",
             "DATE", "PLATE", "ACCOUNT", "EMAIL", "URL"]


def to_opf(row: dict) -> dict:
    """Our row schema -> opf train/eval schema.

    opf wants spans as {"LABEL: surface": [[start, end], ...]}; the same
    label+surface appearing twice just gets two offset pairs.
    """
    spans: dict[str, list[list[int]]] = {}
    for s in row["spans"]:
        key = f'{s["type"]}: {s["text"]}'
        spans.setdefault(key, []).append([s["start"], s["end"]])
    return {
        "text": row["text"],
        "spans": spans,
        "info": {"id": row["id"], "source": row.get("annotator", "")},
    }


def main() -> None:
    rng = random.Random(SEED)
    gold = [json.loads(l) for l in GOLD.open(encoding="utf-8")]
    llm = [json.loads(l) for l in LLM.open(encoding="utf-8")]
    print(f"loaded {len(gold)} gold + {len(llm)} llm-labeled rows")

    # --- carve out 50 test rows from gold, stratified by length bucket ---
    by_bucket: dict[str, list[dict]] = {}
    for r in gold:
        b = r.get("metadata", {}).get("length_bucket", "medium")
        by_bucket.setdefault(b, []).append(r)
    test: list[dict] = []
    for b, rows_b in by_bucket.items():
        k = round(N_TEST * len(rows_b) / len(gold))
        test.extend(rng.sample(rows_b, min(k, len(rows_b))))
    # fix rounding drift to land exactly on N_TEST
    test_ids = {r["id"] for r in test}
    pool_extra = [r for r in gold if r["id"] not in test_ids]
    while len(test) < N_TEST and pool_extra:
        r = pool_extra.pop(rng.randrange(len(pool_extra)))
        test.append(r); test_ids.add(r["id"])
    test = test[:N_TEST]
    test_ids = {r["id"] for r in test}
    print(f"test set: {len(test)} rows (held out of gold, excluded from train/val)")

    # --- train/val pool = remaining gold + all llm-labeled ---
    pool = [r for r in gold if r["id"] not in test_ids] + llm
    rng.shuffle(pool)
    n_val = round(len(pool) * VAL_FRACTION)
    val, train = pool[:n_val], pool[n_val:]
    print(f"train: {len(train)}  |  val: {len(val)}  (from {len(pool)} pooled rows)")

    # --- write ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows_w in [("train", train), ("val", val)]:
        path = OUT_DIR / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in rows_w:
                f.write(json.dumps(to_opf(r), ensure_ascii=False) + "\n")
        spans = sum(len(r["spans"]) for r in rows_w)
        print(f"  wrote {path.relative_to(ROOT)}  ({len(rows_w)} rows, {spans} spans)")

    # test stays in OUR schema — must be human-verified before use
    test_path = OUT_DIR / "test_gold.jsonl"
    with test_path.open("w", encoding="utf-8") as f:
        for r in test:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {test_path.relative_to(ROOT)}  ({len(test)} rows — VERIFY BY HAND)")

    # label space
    label_space = {
        "category_version": "piiclassifier_cn_v1",
        "span_class_names": ["O"] + OUR_TYPES,
    }
    ls_path = OUT_DIR / "label_space.json"
    ls_path.write_text(json.dumps(label_space, ensure_ascii=False, indent=2),
                       encoding="utf-8")
    print(f"  wrote {ls_path.relative_to(ROOT)}  ({len(OUR_TYPES)} labels + O)")

    # --- sanity: label distribution in train ---
    tc = Counter(k.split(":", 1)[0]
                 for r in train for k in to_opf(r)["spans"])
    print(f"\ntrain label distribution: {dict(tc.most_common())}")


if __name__ == "__main__":
    main()
