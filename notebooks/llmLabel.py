"""LLM-label a fresh pool of complaints to expand the training set.

Two phases:

  PHASE 1  (local — needs cache.parquet):
      python notebooks/llmLabel.py --sample-only --n 2000
    Stratified-samples N non-empty complaints (excluding rows already in
    data/gold.jsonl) and writes data/llm_pool.jsonl — a small file.

  PHASE 2  (anywhere with internet — needs the OpenAI API + llm_pool.jsonl):
      python notebooks/llmLabel.py --model gpt-5.4-mini
    Reads data/llm_pool.jsonl, labels each row by calling the OpenAI API
    concurrently (JSON-schema structured output), recovers character offsets
    by exact string search, and writes data/llm_labeled.jsonl.

Needs OPENAI_API_KEY in the environment. No GPU required — phase 2 is just
API calls, so it runs on a laptop or in Colab (CPU runtime is fine).

The 50-row human-verified test set is never touched here — this output is
training/dev data only.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT.parent / "cache.parquet"
GOLD = ROOT / "data" / "gold.jsonl"
POOL = ROOT / "data" / "llm_pool.jsonl"
OUT = ROOT / "data" / "llm_labeled.jsonl"

OUR_TYPES = ["ADDRESS", "PHONE", "ID", "PERSON", "ORG",
             "DATE", "PLATE", "ACCOUNT", "EMAIL", "URL", "SECRET"]

# OpenAI Structured Outputs requires the schema root to be an object, so the
# span list is wrapped under "spans". `strict` mode requires every object to
# set additionalProperties:false and list all properties in `required`.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "spans": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "original_text": {"type": "string"},
                    "type": {"type": "string", "enum": OUR_TYPES},
                },
                "required": ["original_text", "type"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["spans"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You are a professional PII annotator for a Chinese municipal complaint-hotline dataset. Read one complaint record and extract every span of personally identifiable information, following the rules below exactly.

# Output
Return a JSON object {"spans": [...]}. Each span: {"original_text": <verbatim substring>, "type": <one of the 11 types>}. If there is no PII, return {"spans": []}.
CRITICAL: `original_text` MUST be copied character-for-character from the input — no paraphrase, no masking, no added or removed characters. We locate the span by exact string match; a non-verbatim copy is discarded.
Be THOROUGH: almost every complaint names at least one address. Extract an address even when it is woven mid-sentence (e.g. after 在 / 自己是…居民), not only when it follows an obvious cue like 住在 or 地址：.

# The 11 types
- ADDRESS  — a specific place: administrative chain (区/镇/村), road + number, building, unit.
- PHONE    — a personal mobile or landline number.
- ID       — a Chinese national ID (身份证), 15 or 18 digits.
- PERSON   — any named human (family + given name). EVERY named person, public or private — officials included.
- ORG      — a specific PRIVATE organization (company, shop, workplace). NOT government bodies (X局 / X委 / X部 / 政府 / 纪委 / 派出所 / 居委会 / 村委会 / 仲裁), NOT public hospitals or schools, NOT household-name brands (京东, 海底捞, 我爱我家…). Those are hard negatives — do not extract them.
- DATE     — a specific calendar date or date-time. NOT relative time (今天 / 昨天 / 上周).
- PLATE    — a vehicle license plate.
- ACCOUNT  — a bank / payment card or other personal account number. NOT money amounts (2105元 is not an ACCOUNT).
- EMAIL    — an email address.
- URL      — a web URL.
- SECRET   — a password, key, or verification code.

# Never extract
- Case numbers: 兴[2024]-0335006
- Order numbers: 热线-231118-024255, 网络-240729-048117
- Public-service numbers: 12345, 110, 119, 120, 10086
- Pronouns / generic roles: 市民, 自己, 对方, 本人, 老板, 村长, 物业经理
- Relative time: 今天, 昨天, 上周

# Granularity
Extract the minimal sensitive entity, never a full sentence. Each entity gets its own span — never merge an address and an org, or an org and a person, into one span.
- Drop leading verbs / labels: "市民住在大兴区高庄村" → "大兴区高庄村"; "姓名：张新军" → "张新军".
- Drop trailing particles and punctuation.

# Example
Input: 市民反映，自己住在大兴区青云店镇沙堆营村，拖欠人姓名张新军，身份证622628197706173738，老板电话18710106706，要求大兴区住建委处理。
Output:
{"spans":[{"original_text":"大兴区青云店镇沙堆营村","type":"ADDRESS"},
 {"original_text":"张新军","type":"PERSON"},
 {"original_text":"622628197706173738","type":"ID"},
 {"original_text":"18710106706","type":"PHONE"}]}
(大兴区住建委 is a government body → not extracted. 市民 is a pronoun → not extracted.)

# Input complaint
"""


def recover_spans(text: str, items: list[dict]) -> tuple[list[dict], int]:
    """Turn LLM (original_text, type) items into offset spans.

    Returns (spans, n_dropped). A span is dropped when its surface string is
    not found verbatim in the source. Multiple occurrences are placed
    left-to-right, skipping ranges already taken.
    """
    spans: list[dict] = []
    taken: list[tuple[int, int]] = []
    dropped = 0
    for it in items:
        surface = it.get("original_text", "")
        ptype = it.get("type", "")
        if not surface or ptype not in OUR_TYPES:
            dropped += 1
            continue
        starts, i = [], text.find(surface)
        while i != -1:
            starts.append(i)
            i = text.find(surface, i + 1)
        placed = False
        for s in starts:
            e = s + len(surface)
            if any(not (e <= t0 or t1 <= s) for t0, t1 in taken):
                continue
            spans.append({"start": s, "end": e, "text": surface, "type": ptype})
            taken.append((s, e))
            placed = True
            break
        if not placed:
            dropped += 1
    spans.sort(key=lambda x: x["start"])
    return spans, dropped


def length_bucket(desc: str) -> str:
    n = len(desc)
    if n == 0:
        return "empty"
    if n <= 80:
        return "short"
    if n <= 200:
        return "medium"
    if n <= 500:
        return "long"
    return "very_long"


# ---------------------------------------------------------------------------
# Phase 1 — sample the pool (local)
# ---------------------------------------------------------------------------
def sample_pool(n: int, seed: int) -> None:
    import pandas as pd
    if not CACHE.exists():
        raise SystemExit(f"missing {CACHE} — needed for sampling")
    df = pd.read_parquet(CACHE)
    df["EVENT_DESC"] = df["EVENT_DESC"].replace("0", "").fillna("").astype(str)
    used = set()
    if GOLD.exists():
        used = {json.loads(l).get("source_row_index")
                for l in GOLD.open(encoding="utf-8")}
    df = df[~df.index.isin(used)]
    df["lb"] = df["EVENT_DESC"].map(length_bucket)
    df = df[df["lb"] != "empty"]          # empty-text rows teach the NER nothing
    rng = random.Random(seed)
    rows: list[dict] = []
    for _, grp in df.groupby("lb"):
        k = round(n * len(grp) / len(df))
        for i in rng.sample(list(grp.index), min(k, len(grp))):
            rows.append({"source_row_index": int(i),
                         "text": df.at[i, "EVENT_DESC"]})
    POOL.parent.mkdir(parents=True, exist_ok=True)
    with POOL.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Phase 1 done: wrote {len(rows)} rows → {POOL.relative_to(ROOT)} "
          f"({POOL.stat().st_size/1024:.0f} KB).")


# ---------------------------------------------------------------------------
# Phase 2 — label the pool (OpenAI API)
# ---------------------------------------------------------------------------
def label_pool(model_name: str, workers: int) -> None:
    if not POOL.exists():
        raise SystemExit(
            f"missing {POOL.relative_to(ROOT)} — run phase 1 (--sample-only) first.")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("set OPENAI_API_KEY in the environment first.")
    from openai import OpenAI

    raw = [json.loads(l) for l in POOL.open(encoding="utf-8")]
    # Empty-text rows can't carry PII and teach the NER nothing — skip them
    # entirely (also saves an API call each).
    pool = [r for r in raw if r["text"].strip()]
    skipped = len(raw) - len(pool)
    print(f"Labeling {len(pool)} rows with {model_name} "
          f"({skipped} empty-text rows skipped, {workers} workers)")

    # One client, shared across threads (the OpenAI client is thread-safe).
    # max_retries lets the SDK back off automatically on 429 rate limits.
    client = OpenAI(max_retries=8)

    def label_one(idx: int, text: str):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": SYSTEM_PROMPT + text}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "pii_spans", "strict": True,
                                    "schema": RESPONSE_SCHEMA},
                },
                # No temperature / max_tokens: some GPT-5.x models reject custom
                # values for those. Structured output keeps results stable.
            )
            obj = json.loads(resp.choices[0].message.content)
            u = resp.usage
            return idx, obj.get("spans", []), (u.prompt_tokens, u.completion_tokens), None
        except Exception as e:                           # noqa: BLE001
            return idx, [], (0, 0), str(e)

    results: list = [None] * len(pool)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(label_one, i, r["text"]): i for i, r in enumerate(pool)}
        done = 0
        for fut in as_completed(futs):
            idx, items, usage, err = fut.result()
            results[idx] = (items, usage, err)
            done += 1
            if done % 100 == 0 or done == len(pool):
                print(f"  {done}/{len(pool)} ({time.time()-t0:.0f}s)")

    total_spans = total_dropped = errors = written = 0
    in_tok = out_tok = 0
    with OUT.open("w", encoding="utf-8") as f:
        for r, (items, usage, err) in zip(pool, results):
            in_tok += usage[0]
            out_tok += usage[1]
            if err:
                errors += 1
                continue                       # failed rows are NOT written
            spans, dropped = recover_spans(r["text"], items)
            total_spans += len(spans)
            total_dropped += dropped
            written += 1
            f.write(json.dumps({
                "id": f"row_{r['source_row_index']}",
                "source_row_index": r["source_row_index"],
                "text": r["text"],
                "spans": spans,
                "uncertain": False,
                "notes": "",
                "annotator": f"openai:{model_name}",
                "annotated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }, ensure_ascii=False) + "\n")

    print(f"\nPhase 2 done: wrote {written}/{len(pool)} rows → {OUT.relative_to(ROOT)}")
    print(f"  spans kept:    {total_spans}")
    print(f"  spans dropped: {total_dropped}  (surface not found verbatim)")
    print(f"  API errors:    {errors}  (these rows were NOT written)")
    print(f"  tokens:        {in_tok:,} in + {out_tok:,} out "
          f"(estimate cost from current OpenAI pricing)")
    if total_spans + total_dropped:
        rate = total_dropped / (total_spans + total_dropped) * 100
        print(f"  drop rate:     {rate:.1f}%")
    if errors > len(pool) * 0.05:
        print(f"\n  ⚠ {errors}/{len(pool)} rows failed — almost certainly rate "
              f"limiting. Lower --workers (try 2-3) and re-run.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-only", action="store_true",
                    help="phase 1: sample the pool and exit")
    ap.add_argument("--n", type=int, default=2000, help="rows to sample (phase 1)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", help="OpenAI model id for labeling (phase 2)")
    ap.add_argument("--workers", type=int, default=4,
                    help="concurrent API requests (phase 2). New API keys are "
                         "on a low rate-limit tier — keep this small (2-4).")
    args = ap.parse_args()

    if args.sample_only:
        sample_pool(args.n, args.seed)
    elif args.model:
        label_pool(args.model, args.workers)
    else:
        raise SystemExit("pass --sample-only (phase 1) or --model NAME (phase 2)")


if __name__ == "__main__":
    main()
