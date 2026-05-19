# PIIClassifier-CN — Project Plan

## Problem framing

A municipal complaint hotline has ~294k recorded calls / year. Two related problems:

1. **Routing.** Each call must be routed to one of 66 departments (`承办单位`). 68% historically go to 属地 (local district); 32% to specialists.
2. **Analytics.** The same text contains structured information (addresses, complaint types, dates) that could power per-district reports — "this neighborhood has unusually many appliance complaints this month" — but the text is unstructured and contains PII.

PII extraction is the bridge between the two. Extracting addresses cleanly:
- enables anonymization (the privacy half),
- becomes a routing feature (district/township → 属地 office lookup),
- becomes the spatial key for geo-temporal analytics.

## Architecture

A three-tier extraction pipeline feeds two downstream products.

```
raw complaint text
       │
       ▼
┌───────────────────────────────┐
│ Tier 0 — Deterministic        │  regex + Chinese-ID checksum
│   phones, IDs, plates, emails │  ~near-perfect precision
└───────────────────────────────┘
       │
       ▼
┌───────────────────────────────┐
│ Tier 1 — Fine-tuned encoder   │  hfl/chinese-roberta-wwm-ext
│   addresses, names, dates,    │  fine-tuned via LoRA on synthetic
│   accounts, secrets           │  + weak labels + manual gold
│  (OpenAI privacy-filter as    │
│   comparison baseline)        │
└───────────────────────────────┘
       │
       ▼
┌───────────────────────────────┐
│ Tier 2 — Post-processing      │  ID checksum, address parsing,
│   span validation + parsing   │  abstention rule
└──────────────┬────────────────┘
       ┌───────┴───────┐
       ▼               ▼
┌──────────────┐  ┌─────────────────────┐
│ Anonymized   │  │ Structured PII      │
│ text         │  │ {district, township,│
│              │  │  road, ...}         │
└──────┬───────┘  └──────────┬──────────┘
       ▼                     ▼
Product A: Router      Product B: Geo-analytics
TF-IDF + LR on         District/township maps,
anonymized text +      per-area complaint mix,
geo features           temporal anomalies
```

### Why this design

- **Cascade not monolith.** Tier 0 (regex) handles what regex handles perfectly. Tier 1 (model) earns its keep only on fuzzy spans (addresses, names). Bounds compute.
- **Three label sources for Tier 1.** Synthetic data for volume, weak supervision from Tier 0 for realism, manual annotation for a clean test set + a small high-quality train slice.
- **Anonymized text feeds the router.** The router learns from `"[PERSON] 拖欠 [ORG] 工资"` not `"张三 拖欠 嘉涂乐公司 工资"`. Should generalize better, less overfitting. We measure this counterfactually.
- **Address is the bridge.** Same extractor feeds anonymization, routing-as-feature, and geo-analytics. One pipeline, three deliverables.

## Hardware plan

- **Local dev (M4 16GB):** Tier 0, data exploration, annotation tool, evaluation, Streamlit demo. Model backbone: `hfl/chinese-roberta-wwm-ext` (~110M params) via MPS.
- **Colab:** Heavy fine-tuning runs, OpenAI privacy-filter comparison, full-data inference passes when needed.

## Phased plan

### Phase 1 — Foundations (week 1)
- [x] Repo scaffolding
- [ ] Tier 0 module: regex + checksum extractors (phones, IDs, plates, emails, rough addresses)
- [ ] Tier 0 evaluation on the 294k complaint dataset (volume + sample-quality check)
- [ ] Manual gold test set: 200–300 rows with span-level PII labels — locked from training forever
- [ ] Zero-shot baseline: try `openai/privacy-filter` directly on Chinese, measure how bad it is (motivates fine-tuning)

### Phase 2 — Tier 1 model (week 2)
- [ ] Synthetic data generator: Chinese complaint templates × fake names/addresses/IDs → ~10k labeled examples
- [ ] Weak labels: run Tier 0 over all 294k rows, treat outputs as silver labels
- [ ] LoRA fine-tune `chinese-roberta-wwm-ext` on (synthetic + weak + small manual slice)
- [ ] Eval span F1 on the manual gold set; iterate
- [ ] Same fine-tune on Colab with OpenAI privacy-filter as backbone, compare

### Phase 3 — Tier 2 post-processing (week 3 first half)
- [ ] Chinese-ID checksum (already in Tier 0, lift to validation pass)
- [ ] Address parser: span → (district, township, road, building) tuple
- [ ] Geo-resolver: address tuple → Beijing administrative codes
- [ ] Confidence + abstention threshold

### Phase 4 — Anonymized-text routing (week 3 second half)
- [ ] Generate anonymized version of the dataset (PII spans → `[TAG]`)
- [ ] Retrain hierarchical router on anonymized text
- [ ] Add `(district, township)` as features
- [ ] Three-way comparison: raw vs anonymized vs anonymized + geo

### Phase 5 — Geo-analytics (week 4)
- [ ] District/township maps of complaints
- [ ] Per-area complaint-mix view
- [ ] Temporal anomaly detection per district per complaint type (rolling-window z-score)
- [ ] Templated report generation ("Township X had 3.2× normal volume of Y this week")

### Phase 6 — Polish (final stretch)
- [ ] End-to-end notebook walkthrough
- [ ] Streamlit/Gradio demo: paste a complaint, watch it parse → anonymize → route → map
- [ ] Written report

## Evaluation contract

- **Tier 0 + Tier 1 (PII extraction):** span-level F1 on the manual gold test set, per entity type. Token-level precision/recall as supplementary.
- **Anonymization:** % of PII spans correctly redacted; false-redaction rate (over-redaction is also bad).
- **Routing:** top-1 accuracy + macro-F1, time-based train/test split, three-way comparison.
- **Geo-analytics:** qualitative — does it surface anomalies that look real on inspection?

## Decisions log

- **2026-05-13.** Primary backbone is `chinese-roberta-wwm-ext` for local-dev iteration speed; OpenAI privacy-filter is comparison baseline trained on Colab.
- **2026-05-13.** Synthetic + weak + manual is the training-data triad. Manual gold set is the test set, locked.
- **2026-05-13.** Router will be trained on anonymized text as the primary configuration; raw-text router is comparison baseline.
