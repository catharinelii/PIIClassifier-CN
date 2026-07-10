# notebooks/

Standalone pipeline scripts. Run any from the repo root:
`python notebooks/<name>.py`.

These are plain `.py` scripts (not `.ipynb`) so they re-run cleanly and diff in
git. `data/` and `sft/` are git-ignored, so the artifacts below live only on
your machine / Colab Drive.

## Pipeline order

The scripts form one dependency chain — data flows top to bottom.

| # | Script | Stage | What it does | Reads → Writes |
|---|---|---|---|---|
| 1 | `baselineSummary.py` | explore | Run Tier-0 regex extractors over the full 294k complaints; report PII coverage + ID-checksum lift + sample anonymizations. | cache → stdout |
| 2 | `annotationSample.py` | sample | Stratified-sample ~300 rows (length × PII-density, with department/channel/time/hard-negative overlays) for hand-annotation. | cache → `data/to_annotate.jsonl` |
| 3 | `dataAudit.py` | QA | Audit `data/gold.jsonl` — offsets, schema, guide compliance, Tier-0 cross-check. Report only, fixes nothing. Re-run anytime. | `data/gold.jsonl` → stdout |
| 4 | `llmLabel.py` | expand | Phase 1 `--sample-only --n N`: stratified pool → `data/llm_pool.jsonl` (local). Phase 2 `--model NAME`: label the pool via the OpenAI API → `data/llm_labeled.jsonl`. | `data/gold.jsonl`, pool → `data/llm_labeled.jsonl` |
| 5 | `postfilterLabels.py` | clean | Drop the two systematic gpt-5.4-mini precision errors (gov-body ORGs, money ACCOUNTs). Backs up raw once, rewrites in place. | `data/llm_labeled.jsonl` (in place; backup `llm_labeled.raw.jsonl`) |
| 6 | `prepSftData.py` | split | Merge gold + LLM-labeled into the training schema and carve the held-out test set. | `data/gold.jsonl`, `data/llm_labeled.jsonl` → `data/sft/{train,val,test_gold}.jsonl` |
| 7 | `reviewTestSet.py` | verify | `--export` the 50 test rows to Label Studio, then `--import-from` the reviewed export back into `test_gold.jsonl` (Codex draft backed up to `test_gold.codex.jsonl`). | `data/sft/test_gold.jsonl` ↔ Label Studio |
| 8 | `sftOpenmed.py` | train | Full fine-tune from a warm-start checkpoint (default `OpenMed/privacy-filter-multilingual`; flip `--base` for the chinese-roberta rung). Reinits the head to our 41 BIOES labels via HF Trainer. Run on Colab GPU. | `data/sft/{train,val}.jsonl` → `--output-dir` |
| 9 | `evalSft.py` | eval | Strict span-F1 (exact `start,end,type`) of a checkpoint vs. the un-tuned OPF baseline. In-schema (7) + full-schema (10) scopes. | `--test`, `--checkpoint` → stdout |
| 9 | `evalOpenmed.py` | eval | Same strict scoring as `evalSft` (imports it). `--model` works for both zero-shot `OpenMed/...` and our SFT'd checkpoints. | `--test`, `--model` → stdout |

`evalOpenmed.py` imports `score`/`report` from `evalSft.py` — keep both.

## Typical run

```bash
# explore + build the annotation batch
python notebooks/baselineSummary.py
python notebooks/annotationSample.py
# ...hand-annotate in Label Studio → data/gold.jsonl, then audit
python notebooks/dataAudit.py

# expand with weak labels, clean, split
python notebooks/llmLabel.py --sample-only --n 2000
python notebooks/llmLabel.py --model gpt-5.4-mini
python notebooks/postfilterLabels.py
python notebooks/prepSftData.py

# train (Colab GPU) + eval
python notebooks/sftOpenmed.py \
    --train data/sft/train.jsonl --val data/sft/val.jsonl \
    --output-dir /content/drive/MyDrive/piiclassifier_sft/finetuned_multilingual
python notebooks/evalOpenmed.py \
    --test data/sft/test_gold.jsonl --model <output-dir>
```

## Label Studio

Annotation setup and the round-trip (export sample → annotate → import to
`data/gold.jsonl`) are documented in `docs/annotationSetup.md`; the LS labeling
config is `docs/label_studio_config.xml`.
