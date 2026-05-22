"""Zero-shot eval of OpenMed/privacy-filter-multilingual on our test_gold set.

This measures the OpenMed multilingual privacy-filter — itself an `opf train`
fine-tune of openai/privacy-filter on AI4Privacy's 54 categories across 16
languages (incl. zh) — *without any fine-tuning on our data*. The point is to
see how much OpenMed's multilingual SFT alone bought for Chinese PII, before we
warm-start our own SFT on top of it.

Scored with the SAME strict span-F1 as evalSft.py (imported below), so the
numbers line up directly against BASELINE (un-fine-tuned OPF) and our v1:
a prediction is a true positive only if (start, end, type) all match a gold
span exactly.

The model emits 54 ai4privacy categories; we map them onto our 10 types
(OPENMED_TO_OURS). Mapping caveats worth knowing when reading the result:
  - OpenMed decomposes addresses (STREET / BUILDINGNUMBER / CITY / COUNTY /
    STATE / ZIPCODE), whereas our gold uses ONE ADDRESS span. Under strict
    exact-match this systematically *under-credits* ADDRESS (boundaries differ),
    so treat ADDRESS recall here as a lower bound.
  - ID has no direct equivalent; we proxy SSN -> ID. PLATE <- VRM. ACCOUNT <-
    bank/credit/IBAN/masked-number. Several OpenMed types (AGE, AMOUNT, GENDER,
    crypto, IP/MAC, TIME, PREFIX, ...) have no home in our schema and are
    dropped (reported at the end so nothing silently disappears).

The model loads ONCE via OpenMed's torch pipeline (BIOES-Viterbi grouping +
refined char offsets, trust_remote_code), then runs per row. GPU recommended.

    pip install -U "openmed[hf]"
    python notebooks/evalOpenmed.py --test /content/test_gold.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

# Reuse the IDENTICAL scoring code from evalSft so this is apples-to-apples.
# (evalSft imports `opf` only inside main(), so importing it here is cheap and
# does NOT require the opf package to be installed.)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evalSft import score, report, OUR_TYPES, IN_SCHEMA, FULL_SCHEMA  # noqa: E402

MODEL = "OpenMed/privacy-filter-multilingual"

# OpenMed's 54 ai4privacy categories -> our 10 types. Keys are NORMALISED
# (lowercased, non-alphanumerics stripped) so they match whether the pipeline
# emits raw config names (ORGANIZATION, BUILDINGNUMBER, VRM, DATEOFBIRTH) or
# canonical ones (street_address, vehicle_registration, account_number).
OPENMED_TO_OURS = {
    # PERSON
    "firstname": "PERSON", "lastname": "PERSON", "middlename": "PERSON",
    "person": "PERSON",
    # ORG
    "organization": "ORG",
    # ADDRESS  (Western decomposition collapses to our single ADDRESS span)
    "street": "ADDRESS", "streetaddress": "ADDRESS", "buildingnumber": "ADDRESS",
    "secondaryaddress": "ADDRESS", "city": "ADDRESS", "county": "ADDRESS",
    "state": "ADDRESS", "zipcode": "ADDRESS", "location": "ADDRESS",
    # contact
    "phone": "PHONE", "email": "EMAIL", "url": "URL",
    # date
    "date": "DATE", "dateofbirth": "DATE",
    # id  (closest proxy — no Chinese 身份证 category exists)
    "ssn": "ID", "idnum": "ID",
    # account / card
    "accountnumber": "ACCOUNT", "bankaccount": "ACCOUNT", "iban": "ACCOUNT",
    "creditcard": "ACCOUNT", "maskednumber": "ACCOUNT",
    # plate
    "vrm": "PLATE", "vehicleregistration": "PLATE",
}


def norm(label: str) -> str:
    return re.sub(r"[^a-z0-9]", "", label.lower())


def run_openmed(pipe, gold: list[dict], min_conf: float):
    """Predict spans per row. Returns (preds, unmapped_counter, sample)."""
    preds: list[set] = []
    unmapped: collections.Counter = collections.Counter()
    sample: list[tuple] = []  # first non-empty row's raw predictions, for eyeballing
    for r in gold:
        s: set = set()
        if r["text"]:
            items = pipe(r["text"])
            for it in items:
                if float(it.get("score", 1.0)) < min_conf:
                    continue
                raw = it.get("entity_group") or it.get("entity") or ""
                t = OPENMED_TO_OURS.get(norm(raw))
                a, b = int(it["start"]), int(it["end"])
                if not sample and items:
                    sample.append((raw, t, a, b, r["text"][a:b]))
                if t in OUR_TYPES and b > a:
                    s.add((a, b, t))
                elif t is None and raw:
                    unmapped[raw] += 1
        preds.append(s)
    return preds, unmapped, sample


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True, help="verified test_gold.jsonl")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--min-conf", type=float, default=0.5,
                    help="drop predicted spans below this confidence (OpenMed default 0.5)")
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    gold = [json.loads(l) for l in open(args.test, encoding="utf-8")]
    gold_sets = [{(s["start"], s["end"], s["type"]) for s in r["spans"]} for r in gold]
    print(f"loaded {len(gold)} test rows, {sum(len(g) for g in gold_sets)} gold spans")

    print(f"\nloading {MODEL} (loads once) ...")
    try:
        from openmed.torch.privacy_filter import PrivacyFilterTorchPipeline
        pipe = PrivacyFilterTorchPipeline(MODEL, device=args.device)
    except Exception as e:  # fall back to the high-level factory (auto-cuda)
        print(f"  direct load failed ({e}); falling back to create_privacy_filter_pipeline")
        from openmed.core.backends import create_privacy_filter_pipeline
        pipe = create_privacy_filter_pipeline(MODEL)

    print(f"running zero-shot (min-conf={args.min_conf}) ...")
    preds, unmapped, sample = run_openmed(pipe, gold, args.min_conf)

    # Sanity check: show the first row's raw predictions so we can eyeball that
    # labels + offsets are sane before trusting the aggregate numbers.
    if sample:
        print("\nsample raw predictions (first non-empty row):")
        for raw, mapped, a, b, surf in sample:
            print(f"  {raw:>18} -> {str(mapped):>7}  [{a}:{b}]  {surf!r}")

    print("\n" + "=" * 66)
    report("OPENMED-MULTILINGUAL (zero-shot)", gold_sets, preds)

    print("\nOpenMed zero-shot per-type (full-schema):")
    res = score(gold_sets, preds, FULL_SCHEMA)
    for t in OUR_TYPES:
        if t in res["per"]:
            p, r, f = res["per"][t]
            print(f"  {t:>9}: P={p:.3f} R={r:.3f} F1={f:.3f}")
        else:
            print(f"  {t:>9}: (no gold spans in test)")

    if unmapped:
        print("\nunmapped OpenMed labels seen (dropped — no home in our schema):")
        for lab, n in unmapped.most_common():
            print(f"  {lab:>20}: {n}")


if __name__ == "__main__":
    main()
