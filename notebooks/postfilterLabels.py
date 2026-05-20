"""Post-filter LLM-labeled spans: remove the two systematic precision errors
gpt-5.4-mini makes —

  1. gov-body ORGs   — government bureaus / public institutions tagged ORG
                       (per annotationGuide §3.5 these are hard negatives;
                       they also carry routing signal, so must stay un-tagged).
  2. money ACCOUNTs  — money amounts ("9800元") mistyped as ACCOUNT.

Both filters are deterministic rules — no model judgment. Re-runnable on any
LLM-labeled batch in our gold schema. Backs up the raw file once, then writes
the cleaned version in place; prints every dropped span for review.

    python notebooks/postfilterLabels.py
"""
from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "data" / "llm_labeled.jsonl"
RAW_BACKUP = ROOT / "data" / "llm_labeled.raw.jsonl"

# An ORG span is a public body if it contains one of these markers and NO
# private-business marker. "委" subsumes 村委会/居委会/管委会/委员会; "局"
# subsumes every bureau.
GOV_MARKERS = (
    "政府", "局", "委", "街道办", "派出所", "法院", "检察", "仲裁",
    "大队", "支队", "人民医院", "中医院", "卫生院", "卫生服务中心",
    "中学", "小学", "幼儿园", "进修学校",
)
PRIVATE_MARKERS = (
    "公司", "超市", "店", "厂", "酒店", "宾馆", "餐厅", "诊所",
    "事务所", "中介", "合作社", "工作室", "门市",
)
# A real account number is digits-only; any of these means it's a money amount.
MONEY_CHARS = ("元", "块", "万")


def is_gov_body(text: str) -> bool:
    if any(p in text for p in PRIVATE_MARKERS):
        return False
    return any(g in text for g in GOV_MARKERS)


def is_money(text: str) -> bool:
    return any(c in text for c in MONEY_CHARS)


def main() -> None:
    if not TARGET.exists():
        raise SystemExit(f"missing {TARGET.relative_to(ROOT)}")
    rows = [json.loads(l) for l in TARGET.open(encoding="utf-8")]

    dropped_org: Counter = Counter()
    dropped_acct: Counter = Counter()
    for r in rows:
        kept = []
        for s in r["spans"]:
            if s["type"] == "ORG" and is_gov_body(s["text"]):
                dropped_org[s["text"]] += 1
                continue
            if s["type"] == "ACCOUNT" and is_money(s["text"]):
                dropped_acct[s["text"]] += 1
                continue
            kept.append(s)
        r["spans"] = kept

    if not RAW_BACKUP.exists():
        shutil.copy(TARGET, RAW_BACKUP)
        print(f"backed up raw → {RAW_BACKUP.relative_to(ROOT)}")
    with TARGET.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n=== ORG gov-body spans dropped: {sum(dropped_org.values())} "
          f"({len(dropped_org)} unique) ===")
    for t, c in dropped_org.most_common():
        print(f"  {c:>3}x  {t}")
    print(f"\n=== ACCOUNT money-amount spans dropped: {sum(dropped_acct.values())} "
          f"({len(dropped_acct)} unique) ===")
    for t, c in dropped_acct.most_common():
        print(f"  {c:>3}x  {t}")

    remaining = sum(len(r["spans"]) for r in rows)
    print(f"\nremaining spans: {remaining}  → {TARGET.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
