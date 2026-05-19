"""Run Tier-0 extractors over the full 294k dataset and report stats.

This is a script, not a notebook, so we can re-run it cleanly. Convert later
with `jupytext` if we want the notebook form for the writeup.

We answer four questions:
  Q1. How many rows have ≥1 PII span of each type?
  Q2. How much does ID checksum filtering buy us vs. naive 18-digit regex?
  Q3. What % of rows produce SOMETHING (overall PII coverage)?
  Q4. Show 5 fully-anonymized samples so we can eyeball quality.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from piiclassifier_cn.extractors.regex_extractors import (  # noqa: E402
    anonymize,
    extract_all,
    extract_ids,
)
from piiclassifier_cn.spans import PIIType  # noqa: E402

DATA = ROOT.parent / "cache.parquet"
assert DATA.exists(), f"expected parquet cache at {DATA}"

t0 = time.time()
df = pd.read_parquet(DATA)
df["EVENT_DESC"] = df["EVENT_DESC"].replace("0", "")
df["EVENT_DESC"] = df["EVENT_DESC"].fillna("").astype(str)
print(f"Loaded {len(df):,} rows in {time.time()-t0:.1f}s")

t1 = time.time()
df["spans"] = df["EVENT_DESC"].map(extract_all)
print(f"Tier-0 inference on full dataset: {time.time()-t1:.1f}s")

# Q1 — per-type coverage
print("\n=== Q1: per-type row coverage ===")
print(f"{'type':12} {'rows':>10} {'%':>7} {'spans':>10}")
for pt in PIIType:
    rows = df["spans"].map(lambda spans: any(s.type == pt for s in spans)).sum()
    total_spans = df["spans"].map(lambda spans: sum(1 for s in spans if s.type == pt)).sum()
    if rows == 0:
        continue
    print(f"{pt.value:12} {rows:>10,} {rows/len(df)*100:>6.2f}% {total_spans:>10,}")

# Q2 — checksum impact
print("\n=== Q2: ID checksum impact ===")
t2 = time.time()
naive = df["EVENT_DESC"].map(lambda s: extract_ids(s, verify_checksum=False))
strict = df["EVENT_DESC"].map(lambda s: extract_ids(s, verify_checksum=True))
naive_count = naive.map(len).sum()
strict_count = strict.map(len).sum()
print(f"  naive 18-digit regex: {naive_count:,} candidates")
print(f"  with checksum:        {strict_count:,} candidates")
if naive_count:
    print(f"  rejected by checksum: {naive_count - strict_count:,} "
          f"({(naive_count - strict_count) / naive_count * 100:.1f}%)")
print(f"  (took {time.time()-t2:.1f}s)")

# Q3 — overall coverage
print("\n=== Q3: overall PII coverage ===")
any_pii = df["spans"].map(lambda spans: len(spans) > 0)
print(f"  rows with ≥1 PII span:  {any_pii.sum():,} ({any_pii.mean()*100:.2f}%)")
print(f"  rows with 0 PII spans:  {(~any_pii).sum():,} ({(~any_pii).mean()*100:.2f}%)")

# Q4 — eyeball anonymized samples
print("\n=== Q4: 5 anonymized samples ===")
samples = (
    df[any_pii]
      .sample(5, random_state=42)
      [["EVENT_DESC"]]
      .assign(anon=lambda s: s["EVENT_DESC"].map(anonymize))
)
for i, (orig, anon) in enumerate(zip(samples["EVENT_DESC"], samples["anon"]), 1):
    print(f"\n--- sample {i} ---")
    print(f"  ORIG: {orig[:280]}")
    print(f"  ANON: {anon[:280]}")

print(f"\nTotal wall time: {time.time()-t0:.1f}s")
