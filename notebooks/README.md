# notebooks/

Standalone pipeline scripts. Run any from the repo root:
`python notebooks/NN_name.py`.

## Active pipeline

| Script | Stage | What it does |
|---|---|---|
| `01_tier0_baseline.py` | explore | Run the Tier-0 regex extractors over all 294k complaints; report PII coverage. |
| `02_sample_for_annotation.py` | sample | Stratified-sample 300 rows → `data/to_annotate.jsonl`. |
| `05_audit_gold.py` | QA | Audit `data/gold.jsonl` — offsets, schema, guide compliance, Tier-0 cross-check. Re-run anytime. |
| `06_llm_label.py` | expand | Two phases: `--sample-only` builds `data/llm_pool.jsonl` (local); `--model NAME` LLM-labels it via vLLM (Colab GPU) → `data/llm_labeled.jsonl`. |

The Label Studio round-trip (export the sample → annotate → import to
`data/gold.jsonl`) is handled by the converter library
`src/piiclassifier_cn/io/labelstudio.py`. See `docs/annotationSetup.md`.

## archive/

Scripts that already did their one-time job — kept for data lineage, **not
meant to re-run** — plus old run logs:

| File | Note |
|---|---|
| `relabel_org.py` | Re-labeled ORG spans after the guide change. Applied once. |
| `cleanup_gold.py` | Dedup + boundary fixes on the gold draft. Applied once. |
| `eval_openai_privacy_filter.py`, `opf_smoketest.py` | Local OpenAI-model eval — superseded by the Colab workflow. |
| `*.log`, `*.report.txt` | stdout / reports from past runs. |

Archived scripts have stale relative paths — they're frozen references, not
runnable as-is.
