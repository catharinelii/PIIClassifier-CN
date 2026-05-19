"""Pick ~300 rows to hand-annotate, stratified by (length × PII-density)
with overlay constraints for department spread, hard negatives, channel,
and time coverage.

Why this script exists
----------------------
Random sampling 300 rows would give us:
  - ~204 属地 rows (68%), leaving ~96 across the other 65 departments
  - ~50 rows that are EVENT_DESC=="0" (useless to annotate)
  - few or no long-text rows (the Taiwanese-model cliff zone)
  - few hard-negatives (case_no patterns embedded in text, public-service
    phone numbers, government dept references)

So we sample on a grid. Primary axis is text-length bucket (the dominant
failure mode for Chinese NER, per lianghsun/privacy-filter-tw). Secondary
axis is PII density (we want a deliberate mix of high-PII / medium / zero).
Overlay constraints ensure entity-type diversity and adversarial coverage.

Output
------
``data/to_annotate.jsonl`` — one JSON record per row, ready for the annotation
tool. Each record carries Tier-0 regex pre-fills under ``tier0_spans`` so the
tool can surface them as suggestions (which the annotator accepts/edits/rejects).

A short sanity report is printed (and tee'd to ``02_sample_for_annotation.log``
when run as ``python ... | tee ...``).
"""
from __future__ import annotations

import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from piiclassifier_cn.extractors.regex_extractors import extract_all  # noqa: E402
from piiclassifier_cn.spans import Span  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
TOTAL_BUDGET = 300

DATA = ROOT.parent / "cache.parquet"
OUT_DIR = ROOT / "data"
OUT_JSONL = OUT_DIR / "to_annotate.jsonl"
OUT_REPORT = ROOT / "notebooks" / "02_sample_for_annotation.report.txt"

# Length bucket boundaries, in characters of EVENT_DESC (after "0" -> "").
#   empty:     0
#   short:     1-80
#   medium:    81-200
#   long:      201-500
#   very_long: 501+
LENGTH_BUCKETS = ("empty", "short", "medium", "long", "very_long")
PII_DENSITY = ("zero", "medium", "high")

# Target counts per (length × pii_density) cell. Sums to 300.
# Bias: oversample long/very_long and zero-PII in long rows, because:
#   1. The Taiwanese-cliff finding — long text is where models fail.
#   2. Zero-PII in long text is the worst false-positive trap (lots of
#      text where the model could over-fire).
# Empty rows only exist as (empty, zero).
TARGET: dict[tuple[str, str], int] = {
    ("empty", "zero"): 20,
    ("short", "high"): 12,
    ("short", "medium"): 18,
    ("short", "zero"): 15,
    ("medium", "high"): 25,
    ("medium", "medium"): 50,
    ("medium", "zero"): 25,
    ("long", "high"): 35,
    ("long", "medium"): 30,
    ("long", "zero"): 20,
    ("very_long", "high"): 20,
    ("very_long", "medium"): 15,
    ("very_long", "zero"): 15,
}
assert sum(TARGET.values()) == TOTAL_BUDGET, (
    f"TARGET sums to {sum(TARGET.values())}, expected {TOTAL_BUDGET}"
)

# Overlay constraints. After initial cell-sampling, we greedily swap rows
# (within their cell) to satisfy these without disturbing the primary grid.
# Each value is the minimum count required in the final sample.
TOP10_DEPTS = (
    "属地", "市场监管局", "人力社保局", "住建委", "卫生健康委",
    "交通局", "文化和旅游局", "教委", "体育局", "城管委",
)
CONSTRAINTS = {
    "embedded_ref_min":       25,   # rows with case_no/order_num inside EVENT_DESC
    "pub_service_num_min":    15,   # rows mentioning 12345/110/etc.
    "govt_dept_ref_min":      15,   # rows mentioning government dept names
    "channel_min_each":       30,   # for each major channel (热线/网络/...)
    "quarter_min_each":       15,   # for each quarter present in data
    "top10_dept_min_each":    10,   # for each of the top-10 departments
}

# ---------------------------------------------------------------------------
# Feature precomputation
# ---------------------------------------------------------------------------
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


def pii_density(spans: list[Span]) -> str:
    n = len(spans)
    if n == 0:
        return "zero"
    if n <= 2:
        return "medium"
    return "high"


# Embedded ref patterns. Example: "详见热线-231028-004133", "见兴[2023]-0471093".
# These look like PII but must NOT be labeled. Including some in the sample
# trains the model on real adversarial cases.
_EMBEDDED_REF_RE = re.compile(
    r"(?:热线|网络|微信|来访|短信|微博|寄信)-\d{6}-\d{6}"
    r"|兴\[\d{4}\][-‐]\d{7}"
)

# Public-service phone numbers. These appear in complaint bodies ("我打了12345
# 没人接") and the model must learn to skip them.
_PUB_SERVICE_RE = re.compile(
    r"(?<!\d)(?:12345|10086|10010|10000|110|119|120|122|96110|96169)(?!\d)"
)

# Heuristic for government-dept references (these confuse the model into
# tagging them as ORG-of-citizen). Substring match is enough — this is a
# sampling heuristic, not a labeling decision.
_GOVT_PATTERNS = (
    "区政府", "市政府", "政府办公厅", "街道办事处", "街道办",
    "居委会", "村委会", "派出所", "国务院", "市委", "区委",
    "信访办", "纪委", "监察委",
)


def has_embedded_ref(desc: str) -> bool:
    return bool(_EMBEDDED_REF_RE.search(desc))


def has_pub_service_num(desc: str) -> bool:
    return bool(_PUB_SERVICE_RE.search(desc))


def has_govt_dept_ref(desc: str) -> bool:
    return any(p in desc for p in _GOVT_PATTERNS)


def parse_channel(order_num: str) -> str:
    """Pull the channel prefix from an order_num like '热线-231118-024255'.

    Returns the substring before the first '-'. Unknown channels and missing
    order_nums fall back to ``'其他'``.
    """
    if not order_num or "-" not in order_num:
        return "其他"
    return order_num.split("-", 1)[0] or "其他"


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def initial_grid_sample(
    df: pd.DataFrame, rng: random.Random
) -> dict[tuple[str, str], list[int]]:
    """Pick row indices per (length × density) cell up to the cell target.

    If a cell pool is smaller than its target, take everything and log a
    shortfall warning later. Sampling is without replacement.
    """
    cell_picks: dict[tuple[str, str], list[int]] = {}
    for cell, target in TARGET.items():
        pool = df.index[
            (df["length_bucket"] == cell[0]) & (df["pii_density"] == cell[1])
        ].tolist()
        if len(pool) <= target:
            cell_picks[cell] = pool
        else:
            cell_picks[cell] = rng.sample(pool, target)
    return cell_picks


def constraint_count(df: pd.DataFrame, picks: list[int]) -> dict[str, int | dict]:
    """Score the current sample against every overlay constraint."""
    sub = df.loc[picks]
    return {
        "embedded_ref":       int(sub["has_embedded_ref"].sum()),
        "pub_service_num":    int(sub["has_pub_service_num"].sum()),
        "govt_dept_ref":      int(sub["has_govt_dept_ref"].sum()),
        "channels":           dict(Counter(sub["channel"])),
        "quarters":           dict(Counter(sub["quarter"].astype(str))),
        "top10_depts":        {
            d: int((sub["承办单位"] == d).sum()) for d in TOP10_DEPTS
        },
    }


def greedy_constraint_fix(
    df: pd.DataFrame,
    cell_picks: dict[tuple[str, str], list[int]],
    rng: random.Random,
    max_passes: int = 8,
) -> dict[tuple[str, str], list[int]]:
    """Greedy within-cell swaps to satisfy overlay constraints.

    For each unmet constraint we look for a row in the data pool (same cell as
    one of our picks, but not currently picked) that satisfies the constraint,
    and swap out a sample row that doesn't help any unmet constraint. We never
    cross cell boundaries — the primary length×density grid is sacred.
    """
    def has_attr(idx: int, key: str) -> bool:
        if key == "embedded_ref":   return bool(df.at[idx, "has_embedded_ref"])
        if key == "pub_service_num":return bool(df.at[idx, "has_pub_service_num"])
        if key == "govt_dept_ref":  return bool(df.at[idx, "has_govt_dept_ref"])
        if key.startswith("dept:"): return df.at[idx, "承办单位"] == key.split(":", 1)[1]
        if key.startswith("chan:"): return df.at[idx, "channel"] == key.split(":", 1)[1]
        if key.startswith("qtr:"):  return str(df.at[idx, "quarter"]) == key.split(":", 1)[1]
        raise KeyError(key)

    def deficits(picks_flat: list[int]) -> list[str]:
        c = constraint_count(df, picks_flat)
        out: list[str] = []
        if c["embedded_ref"] < CONSTRAINTS["embedded_ref_min"]:
            out.append("embedded_ref")
        if c["pub_service_num"] < CONSTRAINTS["pub_service_num_min"]:
            out.append("pub_service_num")
        if c["govt_dept_ref"] < CONSTRAINTS["govt_dept_ref_min"]:
            out.append("govt_dept_ref")
        for d in TOP10_DEPTS:
            if c["top10_depts"][d] < CONSTRAINTS["top10_dept_min_each"]:
                out.append(f"dept:{d}")
        # Channels: only enforce for channels that have enough source-data to
        # plausibly meet the quota. Skip rare channels.
        ch_pop = df["channel"].value_counts()
        for ch, count in c["channels"].items():
            if ch_pop.get(ch, 0) >= CONSTRAINTS["channel_min_each"] * 3 \
               and count < CONSTRAINTS["channel_min_each"]:
                out.append(f"chan:{ch}")
        # Quarter: same — only enforce where the source has enough rows.
        qt_pop = df["quarter"].astype(str).value_counts()
        for qt, count in c["quarters"].items():
            if qt_pop.get(qt, 0) >= CONSTRAINTS["quarter_min_each"] * 3 \
               and count < CONSTRAINTS["quarter_min_each"]:
                out.append(f"qtr:{qt}")
        return out

    for _ in range(max_passes):
        picks_flat = [i for v in cell_picks.values() for i in v]
        unmet = deficits(picks_flat)
        if not unmet:
            return cell_picks

        progressed = False
        for key in unmet:
            # Find a not-currently-picked row that satisfies this constraint,
            # in some cell where we have a swappable row.
            candidate_pools: dict[tuple[str, str], list[int]] = defaultdict(list)
            picked_set = set(picks_flat)
            for idx in df.index:
                if idx in picked_set:
                    continue
                if not has_attr(idx, key):
                    continue
                cell = (df.at[idx, "length_bucket"], df.at[idx, "pii_density"])
                if cell in cell_picks and cell_picks[cell]:
                    candidate_pools[cell].append(idx)

            if not candidate_pools:
                continue  # truly no rows in any usable cell

            # Pick the cell with the largest candidate pool (most slack).
            cell = max(candidate_pools, key=lambda c: len(candidate_pools[c]))
            new_idx = rng.choice(candidate_pools[cell])
            # Choose a swap-out row in that cell that doesn't itself help any
            # other unmet constraint — minimize collateral damage.
            current = cell_picks[cell]
            scored = sorted(
                current,
                key=lambda i: sum(has_attr(i, k) for k in unmet),
            )
            swap_out = scored[0]
            if has_attr(swap_out, key):
                # Already satisfies; would be a net-zero swap. Skip.
                continue
            cell_picks[cell] = [new_idx if i == swap_out else i for i in current]
            progressed = True

        if not progressed:
            break

    return cell_picks


# ---------------------------------------------------------------------------
# Span -> dict (JSON-friendly)
# ---------------------------------------------------------------------------
def span_to_dict(s: Span) -> dict:
    d = asdict(s)
    d["type"] = s.type.value  # enum -> str
    return d


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    rng = random.Random(SEED)
    t0 = time.time()
    print(f"Reading {DATA} ...")
    df = pd.read_parquet(DATA)
    # EVENT_DESC == "0" is the dataset's missing-value sentinel.
    df["EVENT_DESC"] = df["EVENT_DESC"].replace("0", "").fillna("").astype(str)
    df["EVENT_NAME"] = df["EVENT_NAME"].fillna("").astype(str)
    df["EVENT_TYPE_NAME"] = df["EVENT_TYPE_NAME"].replace("nan", "").fillna("").astype(str)
    df["order_num"] = df["order_num"].fillna("").astype(str)
    df["case_no"] = df["case_no"].fillna("").astype(str)
    print(f"Loaded {len(df):,} rows in {time.time()-t0:.1f}s")

    # --- features --------------------------------------------------------
    t1 = time.time()
    df["length_bucket"] = df["EVENT_DESC"].map(length_bucket)
    df["channel"] = df["order_num"].map(parse_channel)
    df["quarter"] = pd.to_datetime(df["RPT_TIME"]).dt.to_period("Q")
    df["has_embedded_ref"] = df["EVENT_DESC"].map(has_embedded_ref)
    df["has_pub_service_num"] = df["EVENT_DESC"].map(has_pub_service_num)
    df["has_govt_dept_ref"] = df["EVENT_DESC"].map(has_govt_dept_ref)
    print(f"Light features in {time.time()-t1:.1f}s")

    # Run Tier-0 regex once on the full dataset. We use the count for
    # stratification and persist the spans for the annotation tool.
    t2 = time.time()
    print("Running Tier-0 regex over all rows (this takes a few minutes) ...")
    df["tier0_spans"] = df["EVENT_DESC"].map(extract_all)
    df["pii_density"] = df["tier0_spans"].map(pii_density)
    print(f"Tier-0 done in {time.time()-t2:.1f}s")

    # --- pool diagnostics ------------------------------------------------
    print("\n=== Pool sizes (rows available per cell) ===")
    print(f"{'length':>10} {'density':>8} {'n_avail':>10} {'target':>8}")
    for cell in TARGET:
        pool = (
            (df["length_bucket"] == cell[0]) & (df["pii_density"] == cell[1])
        ).sum()
        print(f"{cell[0]:>10} {cell[1]:>8} {pool:>10,} {TARGET[cell]:>8}")

    # --- sample ----------------------------------------------------------
    t3 = time.time()
    cell_picks = initial_grid_sample(df, rng)

    # Deduplicate by case_no across the picked set. Same case = same text,
    # would leak into both train and test if both kept.
    picked_flat = [i for v in cell_picks.values() for i in v]
    seen: set[str] = set()
    deduped_picks: dict[tuple[str, str], list[int]] = {c: [] for c in cell_picks}
    for cell, idxs in cell_picks.items():
        for i in idxs:
            cn = df.at[i, "case_no"]
            if cn and cn in seen:
                continue
            seen.add(cn)
            deduped_picks[cell].append(i)

    # Top up any cell that lost rows to dedup.
    for cell, idxs in deduped_picks.items():
        shortfall = TARGET[cell] - len(idxs)
        if shortfall <= 0:
            continue
        already = set(idxs)
        pool = [
            i for i in df.index[
                (df["length_bucket"] == cell[0])
                & (df["pii_density"] == cell[1])
            ].tolist()
            if i not in already
            and not (df.at[i, "case_no"] in seen and df.at[i, "case_no"])
        ]
        if pool:
            extra = rng.sample(pool, min(shortfall, len(pool)))
            deduped_picks[cell].extend(extra)
            for j in extra:
                seen.add(df.at[j, "case_no"])

    # Constraint pass.
    final_picks = greedy_constraint_fix(df, deduped_picks, rng)
    picks_flat = [i for v in final_picks.values() for i in v]
    print(f"Sampling + constraint fix in {time.time()-t3:.1f}s")

    # --- write JSONL -----------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for i in picks_flat:
            row = df.loc[i]
            rec = {
                "id": f"row_{i}",
                "source_row_index": int(i),
                "case_no": row["case_no"],
                "order_num": row["order_num"],
                "rpt_time": row["RPT_TIME"].isoformat() if pd.notna(row["RPT_TIME"]) else None,
                "event_name": row["EVENT_NAME"],
                "event_desc": row["EVENT_DESC"],
                "event_type_name": row["EVENT_TYPE_NAME"],
                "department": row["承办单位"],
                "stratification": {
                    "length_bucket": row["length_bucket"],
                    "pii_density": row["pii_density"],
                    "channel": row["channel"],
                    "quarter": str(row["quarter"]),
                    "has_embedded_ref": bool(row["has_embedded_ref"]),
                    "has_pub_service_num": bool(row["has_pub_service_num"]),
                    "has_govt_dept_ref": bool(row["has_govt_dept_ref"]),
                },
                "tier0_spans": [span_to_dict(s) for s in row["tier0_spans"]],
                "annotated": False,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(picks_flat):,} rows to {OUT_JSONL.relative_to(ROOT)}")

    # --- sanity report ---------------------------------------------------
    lines = []
    def out(s: str = "") -> None:
        print(s)
        lines.append(s)

    out("\n" + "=" * 60)
    out("Sanity report")
    out("=" * 60)
    out(f"Total picked: {len(picks_flat)}")
    out(f"Total budget: {TOTAL_BUDGET}")
    out(f"Seed: {SEED}")

    out("\nPer-cell achieved vs target:")
    out(f"  {'length':>10} {'density':>8} {'target':>8} {'achieved':>10} {'note':>10}")
    for cell, target in TARGET.items():
        achieved = len(final_picks.get(cell, []))
        note = "" if achieved >= target else "SHORTFALL"
        out(f"  {cell[0]:>10} {cell[1]:>8} {target:>8} {achieved:>10} {note:>10}")

    c = constraint_count(df, picks_flat)
    out("\nConstraint satisfaction:")
    out(f"  embedded_ref (target ≥{CONSTRAINTS['embedded_ref_min']}): {c['embedded_ref']}")
    out(f"  pub_service_num (target ≥{CONSTRAINTS['pub_service_num_min']}): {c['pub_service_num']}")
    out(f"  govt_dept_ref (target ≥{CONSTRAINTS['govt_dept_ref_min']}): {c['govt_dept_ref']}")
    out(f"  top-10 dept coverage (target ≥{CONSTRAINTS['top10_dept_min_each']} each):")
    for d in TOP10_DEPTS:
        cnt = c["top10_depts"][d]
        flag = "" if cnt >= CONSTRAINTS["top10_dept_min_each"] else "  ⚠ short"
        out(f"      {d:>10}: {cnt}{flag}")
    out("  channel distribution:")
    for ch, n in sorted(c["channels"].items(), key=lambda x: -x[1]):
        out(f"      {ch:>10}: {n}")
    out("  quarter distribution:")
    for qt, n in sorted(c["quarters"].items()):
        out(f"      {qt:>10}: {n}")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    out(f"\nReport saved to {OUT_REPORT.relative_to(ROOT)}")
    out(f"Total wall time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
