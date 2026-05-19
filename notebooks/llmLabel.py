"""LLM-label a fresh pool of complaints to expand the training set.

Two phases — run them on different machines:

  PHASE 1  (local, no GPU — needs cache.parquet):
      python notebooks/06_llm_label.py --sample-only --n 2000
    Stratified-samples N complaints (excluding rows already in
    data/gold.jsonl) and writes data/llm_pool.jsonl — a small file.

  PHASE 2  (Colab GPU — needs vLLM + data/llm_pool.jsonl):
      python notebooks/06_llm_label.py --model Qwen/Qwen3-8B
    Reads data/llm_pool.jsonl, labels each row with vLLM (schema-enforced
    JSON), recovers character offsets by exact string search, and writes
    data/llm_labeled.jsonl in our gold schema.

The 50-row human-verified test set is never touched here — this output is
training/dev data only.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT.parent / "cache.parquet"
GOLD = ROOT / "data" / "gold.jsonl"
POOL = ROOT / "data" / "llm_pool.jsonl"
OUT = ROOT / "data" / "llm_labeled.jsonl"

OUR_TYPES = ["ADDRESS", "PHONE", "ID", "PERSON", "ORG",
             "DATE", "PLATE", "ACCOUNT", "EMAIL", "URL", "SECRET"]

# Structured-output schema: LLM emits surface string + one of our 11 types.
# Deliberately NO start/end — LLMs hallucinate offsets; we recover them below.
PRIVACY_SCHEMA = {
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
}

SYSTEM_PROMPT = """You are a professional PII annotator for a Chinese municipal complaint-hotline dataset. Read one complaint record and extract every span of personally identifiable information, following the rules below exactly.

# Output
Return a JSON array. Each item: {"original_text": <verbatim substring>, "type": <one of the 11 types>}. If there is no PII, return [].
CRITICAL: `original_text` MUST be copied character-for-character from the input — no paraphrase, no masking, no added or removed characters. We locate the span by exact string match; a non-verbatim copy is discarded.

# The 11 types
- ADDRESS  — a specific place: administrative chain (区/镇/村), road + number, building, unit.
- PHONE    — a personal mobile or landline number.
- ID       — a Chinese national ID (身份证), 15 or 18 digits.
- PERSON   — any named human (family + given name). EVERY named person, public or private — officials included.
- ORG      — a specific PRIVATE organization (company, shop, workplace). NOT government bodies (X局 / X委 / X部 / 政府 / 纪委 / 派出所 / 居委会 / 村委会), NOT public hospitals or schools, NOT household-name brands (京东, 海底捞, 我爱我家…). Those are hard negatives — do not extract them.
- DATE     — a specific calendar date or date-time. NOT relative time (今天 / 昨天 / 上周).
- PLATE    — a vehicle license plate.
- ACCOUNT  — a bank / payment card or other personal account number.
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
Extract the minimal sensitive entity, never a full sentence.
- Drop leading verbs / labels: "市民住在大兴区高庄村" → "大兴区高庄村"; "姓名：张新军" → "张新军".
- Drop trailing particles and punctuation.

# Example
Input: 市民反映，自己住在大兴区青云店镇沙堆营村，拖欠人姓名张新军，身份证622628197706173738，老板电话18710106706，要求大兴区住建委处理。
Output:
[{"original_text":"大兴区青云店镇沙堆营村","type":"ADDRESS"},
 {"original_text":"张新军","type":"PERSON"},
 {"original_text":"622628197706173738","type":"ID"},
 {"original_text":"18710106706","type":"PHONE"}]
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
    size_kb = POOL.stat().st_size / 1024
    print(f"Phase 1 done: wrote {len(rows)} rows → {POOL.relative_to(ROOT)} "
          f"({size_kb:.0f} KB). Upload this file to Colab for phase 2.")


# ---------------------------------------------------------------------------
# Phase 2 — label the pool (Colab GPU)
# ---------------------------------------------------------------------------
def label_pool(model_name: str) -> None:
    if not POOL.exists():
        raise SystemExit(
            f"missing {POOL.relative_to(ROOT)} — run phase 1 (--sample-only) "
            f"locally first, then upload the file here.")
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    from transformers import AutoTokenizer

    pool = [json.loads(l) for l in POOL.open(encoding="utf-8")]
    print(f"Labeling {len(pool)} rows with {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sampling = SamplingParams(
        temperature=0.1, top_p=0.1, repetition_penalty=1.05, max_tokens=4096,
        structured_outputs=StructuredOutputsParams(json=PRIVACY_SCHEMA),
    )
    llm = LLM(model=model_name, dtype="float16", gpu_memory_utilization=0.9)

    prompts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": SYSTEM_PROMPT + r["text"]}],
        tokenize=False, add_generation_prompt=True,
        enable_thinking=False)            # enable_thinking flag is Qwen3-specific
        for r in pool]

    t0 = time.time()
    outputs = llm.generate(prompts, sampling)
    print(f"vLLM generate: {time.time()-t0:.0f}s")

    total_spans = total_dropped = parse_fail = 0
    with OUT.open("w", encoding="utf-8") as f:
        for r, out in zip(pool, outputs):
            raw = out.outputs[0].text.strip()
            try:
                items = json.loads(raw)
            except json.JSONDecodeError:
                items, parse_fail = [], parse_fail + 1
            spans, dropped = recover_spans(r["text"], items)
            total_spans += len(spans)
            total_dropped += dropped
            f.write(json.dumps({
                "id": f"row_{r['source_row_index']}",
                "source_row_index": r["source_row_index"],
                "text": r["text"],
                "spans": spans,
                "uncertain": False,
                "notes": "",
                "annotator": f"llm:{model_name}",
                "annotated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }, ensure_ascii=False) + "\n")

    print(f"\nPhase 2 done: wrote {len(pool)} rows → {OUT.relative_to(ROOT)}")
    print(f"  spans kept:    {total_spans}")
    print(f"  spans dropped: {total_dropped}  (surface not found verbatim)")
    print(f"  parse fails:   {parse_fail}")
    if total_spans + total_dropped:
        rate = total_dropped / (total_spans + total_dropped) * 100
        print(f"  drop rate:     {rate:.1f}%  (>10% → tighten prompt / bigger model)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-only", action="store_true",
                    help="phase 1: sample the pool and exit")
    ap.add_argument("--n", type=int, default=2000, help="rows to sample (phase 1)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", help="HF model id for vLLM (phase 2)")
    args = ap.parse_args()

    if args.sample_only:
        sample_pool(args.n, args.seed)
    elif args.model:
        label_pool(args.model)
    else:
        raise SystemExit("pass --sample-only (phase 1) or --model NAME (phase 2)")


if __name__ == "__main__":
    main()
