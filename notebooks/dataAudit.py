"""Quality audit of data/gold.jsonl before we trust it as gold-standard data.

Checks, in order of severity:
  A. Structural integrity — schema, span offsets, types.
  B. Annotation-guide compliance — case numbers, public-service numbers,
     pronouns must NOT be labeled; addresses must not include leading verbs.
  C. Cross-check vs Tier-0 regex — for the entity types regex is near-perfect
     on (checksum-valid IDs, well-formed mobiles), did gold catch them?
  D. Distributional sanity — label counts, coverage, uncertain usage.

This script does NOT fix anything. It produces a report so we can decide
whether the data clears the bar — and if not, exactly what to send back.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from piiclassifier_cn.extractors.regex_extractors import (  # noqa: E402
    extract_ids, extract_phones, extract_all,
)
from piiclassifier_cn.spans import PIIType  # noqa: E402

GOLD = ROOT / "data" / "gold.jsonl"
VALID_TYPES = {t.value for t in PIIType}

# Patterns that the annotation guide says must NEVER be labeled.
import re
CASE_NO_RE = re.compile(r"兴\[\d{4}\][-‐]\d{7}")
ORDER_NO_RE = re.compile(r"(?:热线|网络|微信|来访|短信|微博|寄信)-\d{6}-\d{6}")
PUB_SERVICE = {"12345", "10086", "10010", "10000", "110", "119", "120",
               "122", "96110", "96169"}
ADDR_LEADING_VERBS = ("住在", "位于", "来自", "在")


def main() -> None:
    rows = [json.loads(l) for l in GOLD.open(encoding="utf-8")]
    print(f"Loaded {len(rows)} rows from {GOLD.relative_to(ROOT)}\n")

    # Severity buckets.
    errors: list[str] = []     # disqualifying — must fix
    warnings: list[str] = []   # suspicious — review needed
    notes: list[str] = []      # informational

    # ---- A. Structural integrity ----------------------------------------
    span_total = 0
    for r in rows:
        rid = r.get("id", "?")
        text = r.get("text", "")
        if "spans" not in r:
            errors.append(f"{rid}: missing 'spans' field")
            continue
        for j, s in enumerate(r["spans"]):
            span_total += 1
            for key in ("start", "end", "type", "text"):
                if key not in s:
                    errors.append(f"{rid} span#{j}: missing '{key}'")
            if "start" not in s or "end" not in s:
                continue
            st, en = s["start"], s["end"]
            # offset bounds
            if not (0 <= st < en <= len(text)):
                errors.append(
                    f"{rid} span#{j}: bad offsets start={st} end={en} "
                    f"(text len {len(text)})")
                continue
            # surface form must match the offsets
            if text[st:en] != s.get("text"):
                errors.append(
                    f"{rid} span#{j}: text mismatch — offsets give "
                    f"{text[st:en]!r} but span.text is {s.get('text')!r}")
            # type must be in our schema
            if s.get("type") not in VALID_TYPES:
                errors.append(f"{rid} span#{j}: unknown type {s.get('type')!r}")
        # empty text must have no spans
        if text == "" and r["spans"]:
            errors.append(f"{rid}: empty text but {len(r['spans'])} spans")
        # overlapping spans within a row
        ss = sorted(r["spans"], key=lambda s: s.get("start", 0))
        for a, b in zip(ss, ss[1:]):
            if a.get("end", 0) > b.get("start", 0):
                warnings.append(
                    f"{rid}: overlapping spans {a.get('text')!r} / {b.get('text')!r}")

    # ---- B. Annotation-guide compliance ---------------------------------
    for r in rows:
        rid = r["id"]
        text = r.get("text", "")
        spanset = {(s["start"], s["end"]) for s in r["spans"] if "start" in s}
        for s in r["spans"]:
            stext = s.get("text", "")
            stype = s.get("type")
            # case / order numbers must not be labeled
            if CASE_NO_RE.fullmatch(stext) or ORDER_NO_RE.fullmatch(stext):
                errors.append(f"{rid}: labeled a case/order number {stext!r} as {stype}")
            # case/order number *inside* a span
            if CASE_NO_RE.search(stext) or ORDER_NO_RE.search(stext):
                warnings.append(f"{rid}: span {stext!r} ({stype}) contains a case/order number")
            # public-service numbers must not be PHONE
            if stype == "PHONE" and stext in PUB_SERVICE:
                errors.append(f"{rid}: labeled public-service number {stext!r} as PHONE")
            # address leading-verb violation
            if stype == "ADDRESS":
                for v in ADDR_LEADING_VERBS:
                    if stext.startswith(v):
                        warnings.append(f"{rid}: ADDRESS span starts with verb {v!r}: {stext!r}")
                        break
            # leading/trailing whitespace or punctuation
            if stext != stext.strip() or stext.strip("，。：；、（）\"' ") != stext:
                warnings.append(f"{rid}: span {stext!r} ({stype}) has stray edge punctuation/space")

    # ---- C. Cross-check vs Tier-0 regex ---------------------------------
    # Regex is near-perfect on checksum-valid IDs and well-formed mobiles.
    # If gold missed one of those, it's a recall miss.
    id_misses = phone_misses = 0
    for r in rows:
        text = r.get("text", "")
        if not text:
            continue
        gold_offsets = {(s["start"], s["end"]) for s in r["spans"] if "start" in s}
        gold_by_offset = {(s["start"], s["end"]): s["type"]
                          for s in r["spans"] if "start" in s}
        for rid_span in extract_ids(text, verify_checksum=True):
            key = (rid_span.start, rid_span.end)
            if key not in gold_offsets:
                id_misses += 1
                errors.append(
                    f"{r['id']}: gold missed checksum-valid ID "
                    f"{rid_span.text!r} that regex found")
            elif gold_by_offset.get(key) != "ID":
                warnings.append(
                    f"{r['id']}: checksum-valid ID {rid_span.text!r} labeled "
                    f"as {gold_by_offset.get(key)}, expected ID")
        for ph in extract_phones(text):
            key = (ph.start, ph.end)
            if key not in gold_offsets:
                phone_misses += 1
                warnings.append(
                    f"{r['id']}: gold missed mobile {ph.text!r} that regex found")

    # ---- D. Distributional sanity ---------------------------------------
    type_counts = Counter(s["type"] for r in rows for s in r["spans"])
    rows_with_spans = sum(1 for r in rows if r["spans"])
    empty_rows = sum(1 for r in rows if r.get("text", "") == "")
    uncertain = sum(1 for r in rows if r.get("uncertain"))
    with_notes = sum(1 for r in rows if r.get("notes"))
    annotators = Counter(r.get("annotator", "") for r in rows)
    spans_per_row = [len(r["spans"]) for r in rows]

    # ---- Report ---------------------------------------------------------
    print("=" * 64)
    print("GOLD AUDIT REPORT")
    print("=" * 64)

    print(f"\n[D] Distribution")
    print(f"  rows:               {len(rows)}")
    print(f"  empty-text rows:    {empty_rows}")
    print(f"  rows with ≥1 span:  {rows_with_spans} ({rows_with_spans/len(rows)*100:.1f}%)")
    print(f"  total spans:        {span_total}")
    print(f"  spans/row:          min={min(spans_per_row)} "
          f"max={max(spans_per_row)} mean={sum(spans_per_row)/len(rows):.2f}")
    print(f"  uncertain flagged:  {uncertain}")
    print(f"  rows with notes:    {with_notes}")
    print(f"  annotator field:    {dict(annotators)}")
    print(f"  span types:")
    for t, c in type_counts.most_common():
        print(f"      {t:>9}: {c}")
    missing_types = VALID_TYPES - set(type_counts)
    if missing_types:
        print(f"  types never used:   {sorted(missing_types)}")

    print(f"\n[C] Tier-0 cross-check")
    print(f"  checksum-valid IDs gold missed:  {id_misses}")
    print(f"  well-formed mobiles gold missed: {phone_misses}")

    print(f"\n{'='*64}")
    print(f"ERRORS (disqualifying): {len(errors)}")
    for e in errors[:40]:
        print(f"  ✗ {e}")
    if len(errors) > 40:
        print(f"  ... and {len(errors)-40} more")

    print(f"\nWARNINGS (review needed): {len(warnings)}")
    for w in warnings[:40]:
        print(f"  ⚠ {w}")
    if len(warnings) > 40:
        print(f"  ... and {len(warnings)-40} more")

    print(f"\n{'='*64}")
    verdict = "PASS" if not errors else "FAIL"
    print(f"VERDICT: {verdict}  ({len(errors)} errors, {len(warnings)} warnings)")


if __name__ == "__main__":
    main()
