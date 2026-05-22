# PIIClassifier-CN — Eval Results: v1 vs. Baseline

*Last updated 2026-05-22. Scored by `notebooks/evalSft.py` on `data/sft/test_gold.jsonl` (50 human-verified rows, 174 gold spans, 44/50 rows containing PII).*

---

## 0. TL;DR

Fine-tuning OPF tripled span-F1 (full-schema **0.117 → 0.439**, ≈3.8×), and almost all of the gain is **recall**. But the result splits cleanly in two: **structured types** (PHONE/ID/URL) are nearly solved (F1 > 0.8), while **semantic types** (ADDRESS/ORG/PERSON) lag — because OPF's English byte-BPE tokenizer fragments Chinese characters. That ceiling is exactly what a Chinese-native backbone would attack next.

---

## 1. What was measured

**Metric — strict span-F1.** A prediction is a true positive only if `(start, end, type)` *all* match a gold span exactly. No partial credit for boundary slips or wrong types.

**Two models:**
- **Baseline** — OPF *un*-fine-tuned, with its 7 native English labels mapped onto ours (`private_address`→ADDRESS, etc.). The "before fine-tuning" picture.
- **v1** — our fine-tuned checkpoint (`sft/finetuned_checkpoint/`), emitting our 10 labels natively.

**Two scopes:**
- **in-schema (7 types)** — only the types the base model *can* produce (ADDRESS/PHONE/PERSON/EMAIL/URL/DATE/ACCOUNT). A fair home-turf comparison.
- **full-schema (10 types)** — our whole ask. The base scores 0 on ID/PLATE/ORG by construction — it has no such categories, so 48 of the 174 gold spans are unreachable for it.

---

## 2. Headline numbers

| Scope | Model | Precision | Recall | micro-F1 | macro-F1 | TP / FP / FN |
|---|---|---|---|---|---|---|
| in-schema (7) | Baseline | 0.265 | 0.103 | **0.149** | 0.209 | 13 / 36 / 113 |
| in-schema (7) | **v1** | 0.619 | 0.413 | **0.495** | 0.534 | 52 / 32 / 74 |
| full-schema (10) | Baseline | 0.265 | 0.075 | **0.117** | 0.139 | 13 / 36 / 161 |
| full-schema (10) | **v1** | 0.587 | 0.351 | **0.439** | 0.470 | 61 / 43 / 113 |

- **Fine-tuning gives a 3.3–3.8× jump in F1.** Full-schema 0.117 → 0.439 (3.75×); in-schema 0.149 → 0.495 (3.3×).
- **The gain is almost entirely recall.** The baseline finds only 13 of 174 spans (7.5% recall). It isn't especially *wrong* when it fires (precision 0.265) — it just barely fires on Chinese text. v1 lifts full-schema recall to 0.351 (≈4.7×) and precision to 0.587 (≈2.2×).

---

## 3. Per-type breakdown (v1, full-schema)

Sorted by support (number of gold spans of that type in the test set).

| Type | Support | Precision | Recall | F1 | Read |
|---|---|---|---|---|---|
| ADDRESS | 62 | 0.500 | 0.435 | 0.466 | semantic — biggest drag on micro-F1 |
| ORG | 44 | 0.412 | 0.159 | 0.230 | semantic — weakest type |
| DATE | 37 | 0.737 | 0.378 | 0.500 | mixed |
| PERSON | 17 | 1.000 | 0.235 | 0.381 | semantic — precise but timid |
| PHONE | 8 | 1.000 | 0.750 | 0.857 | structured — strong |
| PLATE | 2 | 0.000 | 0.000 | 0.000 | n=2, anecdotal |
| ID | 2 | 0.667 | 1.000 | 0.800 | n=2, anecdotal |
| URL | 1 | 1.000 | 1.000 | 1.000 | n=1, anecdotal |
| ACCOUNT | 1 | 0.000 | 0.000 | 0.000 | n=1, anecdotal |
| EMAIL | 0 | — | — | — | no gold spans in test |

---

## 4. Interpretation

### 4.1 Structured vs. semantic — the core finding
- **Structured types win** — URL 1.0, PHONE 0.857, ID 0.800. These have rigid surface patterns (digits, `http`, the 18-char ID checksum). The signal lives in the *bytes*, so a broken tokenizer barely hurts.
- **Semantic types lag** — ORG 0.230, PERSON 0.381, ADDRESS 0.466. These require understanding Chinese *meaning and context* — exactly what an English byte-BPE tokenizer is worst at. They are also our three highest-support types (123 of 174 gold spans = **71% of the test**), so they set the micro-F1 ceiling.
- **PERSON is the cleanest illustration:** precision 1.0, recall 0.235. When v1 commits to a name it is *always* right, but it misses 3 of every 4 — classic under-firing from a tokenizer that can't segment Chinese names.

### 4.2 Why the gain is recall, not precision
OPF's byte-BPE tokenizer was trained on English; Chinese characters arrive as broken byte fragments, so the base model effectively can't "see" Chinese entities and stays silent (7.5% recall). SFT teaches it to recover entities from those fragments — which shows up as recall climbing ~4–5×.

### 4.3 micro vs. macro, and why v1's macro (0.470) > micro (0.439)
- **micro-F1** pools every span, so it is dominated by the frequent types (ADDRESS/ORG/DATE).
- **macro-F1** averages the per-type F1s with equal weight, so the rare structured types it nails (URL, PHONE, ID) pull it up.
- macro > micro therefore says: **v1 is better at rare structured types than at common semantic ones.**

### 4.4 in-schema vs. full-schema gap
The baseline drops 0.149 → 0.117 across scopes because 3 of our types (ID, PLATE, ORG) are structurally impossible for it. v1 only drops 0.495 → 0.439, meaning it *does* learn the 3 new types to a degree (ID strong, ORG partial, PLATE not yet).

### 4.5 Honesty caveats
- PLATE/ID/URL/ACCOUNT each have only 1–2 gold spans. ID=0.800 and URL=1.0 look great but rest on 2 and 1 spans respectively — do not over-claim. The trustworthy per-type numbers are ADDRESS, ORG, DATE, PERSON (plus the pooled micro-F1).
- Test set is 50 rows. Treat these as directional, not production-grade.

---

## 5. So what (next rung)

This eval *quantifies* why OPF is a strong baseline but not the endgame for Chinese PII: the structured half is essentially solved on OPF, and all remaining headroom is in the semantic types (ADDRESS/ORG/PERSON), which are gated by the tokenizer. A char-level Chinese-native backbone (`hfl/chinese-roberta-wwm-ext`) attacks exactly that bottleneck. The honest framing is not "OPF failed" but:

> **OPF fine-tuning solved the easy half and proved the hard half is tokenizer-bound — which is why the next rung uses a Chinese-native tokenizer.**

*v2 (5 epochs, SECRET removed) was attempted but the Colab training run stalled at the epoch-2 boundary under memory pressure; v1 is the reported result.*
