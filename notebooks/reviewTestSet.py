"""Review the 50 gold test rows via Label Studio.

Two modes:

  --export                     read data/sft/test_gold.jsonl, write
                               data/sft/test_for_labelstudio.json (LS
                               import file). Drop it into a fresh LS project.

  --import-from <EXPORT.json>  after you've reviewed in LS and Project →
                               Export → JSON, point this at the file; we
                               overwrite data/sft/test_gold.jsonl with the
                               verified labels (Codex draft is backed up to
                               data/sft/test_gold.codex.jsonl on first run).

The pre-existing Codex labels appear in LS as colored *suggestions* (a
"predictions" block) — accept / edit / reject each one. Don't rubber-stamp:
the whole point of this set is to be an independent ground truth.
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SFT = ROOT / "data" / "sft"
INPUT = SFT / "test_gold.jsonl"
LS_IMPORT = SFT / "test_for_labelstudio.json"
BACKUP = SFT / "test_gold.codex.jsonl"
CONFIG_XML = ROOT / "docs" / "label_studio_config.xml"

UNCERTAIN_CHOICE = "Uncertain — flag for review"
VALID_TYPES = {"ADDRESS", "PHONE", "ID", "PERSON", "ORG", "DATE", "PLATE",
               "ACCOUNT", "EMAIL", "URL"}


# ---------------------------------------------------------------------------
# our gold schema -> Label Studio import tasks
# ---------------------------------------------------------------------------
def to_ls_tasks(rows: list[dict]) -> list[dict]:
    tasks = []
    for r in rows:
        text = r["text"]
        prefills = []
        for s in r.get("spans", []):
            prefills.append({
                "from_name": "label", "to_name": "text", "type": "labels",
                "value": {
                    "start": int(s["start"]),
                    "end": int(s["end"]),
                    "text": s.get("text", text[s["start"]:s["end"]]),
                    "labels": [s["type"]],
                },
            })
        meta = r.get("metadata", {})
        task: dict = {
            "data": {
                "id": r["id"],
                "text": text,
                "department": meta.get("department", ""),
                "length_bucket": meta.get("length_bucket", ""),
                "channel": meta.get("channel", ""),
                "case_no": meta.get("case_no", ""),
                "order_num": meta.get("order_num", ""),
                "rpt_time": meta.get("rpt_time", ""),
                "event_name": meta.get("event_name", ""),
                "event_type_name": meta.get("event_type_name", ""),
                "source_row_index": r.get("source_row_index"),
            },
        }
        if prefills:
            task["predictions"] = [{
                "model_version": "codex_draft",
                "result": prefills,
            }]
        tasks.append(task)
    return tasks


# ---------------------------------------------------------------------------
# Label Studio export -> verified gold rows
# ---------------------------------------------------------------------------
def from_ls_export(items: list[dict], annotator: str) -> list[dict]:
    out: list[dict] = []
    for item in items:
        data = item.get("data", {})
        text = data.get("text", "")
        for ann in item.get("annotations") or []:
            if ann.get("was_cancelled"):
                continue
            spans: list[dict] = []
            uncertain = False
            notes = ""
            for r in ann.get("result", []):
                rtype = r.get("type")
                val = r.get("value", {})
                if rtype == "labels":
                    labels = val.get("labels") or []
                    if not labels or labels[0] not in VALID_TYPES:
                        continue
                    span_text = val.get("text") or text[val["start"]:val["end"]]
                    spans.append({
                        "start": int(val["start"]),
                        "end": int(val["end"]),
                        "text": span_text,
                        "type": labels[0],
                    })
                elif rtype == "choices":
                    if UNCERTAIN_CHOICE in (val.get("choices") or []):
                        uncertain = True
                elif rtype == "textarea":
                    texts = val.get("text") or []
                    notes = ("\n".join(texts) if isinstance(texts, list)
                             else str(texts))
            spans.sort(key=lambda s: (s["start"], s["end"]))
            out.append({
                "id": data.get("id"),
                "source_row_index": data.get("source_row_index"),
                "text": text,
                "spans": spans,
                "uncertain": uncertain,
                "notes": notes.strip(),
                "annotator": f"{annotator} (verified)",
                "annotated_at": (ann.get("updated_at")
                                 or ann.get("created_at")
                                 or datetime.utcnow().isoformat()),
                "metadata": {
                    "department": data.get("department", ""),
                    "length_bucket": data.get("length_bucket", ""),
                    "channel": data.get("channel", ""),
                    "case_no": data.get("case_no", ""),
                    "order_num": data.get("order_num", ""),
                    "rpt_time": data.get("rpt_time", ""),
                    "event_name": data.get("event_name", ""),
                    "event_type_name": data.get("event_type_name", ""),
                },
            })
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    ap.add_argument("--export", action="store_true",
                    help="write the LS import file from test_gold.jsonl")
    ap.add_argument("--import-from", dest="import_from", metavar="EXPORT.json",
                    help="LS export JSON to ingest -> verified test_gold.jsonl")
    ap.add_argument("--annotator", default="catharine",
                    help="annotator name recorded in the verified file")
    args = ap.parse_args()

    if args.export:
        if not INPUT.exists():
            raise SystemExit(f"missing {INPUT.relative_to(ROOT)} — "
                             f"run prepSftData.py first")
        rows = [json.loads(l) for l in INPUT.open(encoding="utf-8")]
        tasks = to_ls_tasks(rows)
        LS_IMPORT.write_text(json.dumps(tasks, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        prefills = sum(1 for t in tasks if t.get("predictions"))
        spans = sum(len(t.get("predictions", [{}])[0].get("result", []))
                    for t in tasks if t.get("predictions"))
        print(f"wrote {LS_IMPORT.relative_to(ROOT)}  "
              f"({len(tasks)} tasks; {prefills} have Codex pre-fills "
              f"= {spans} suggested spans)")
        print(f"\nNext steps:")
        print("  1. label-studio start    # in a terminal")
        print("  2. Create a NEW project (don't reuse the old 300-row one).")
        print(f"  3. Settings → Labeling Interface → Code → paste the contents")
        print(f"     of {CONFIG_XML.relative_to(ROOT)}.")
        print(f"  4. Project → Import → upload {LS_IMPORT.relative_to(ROOT)}.")
        print("  5. Review every row. Don't rubber-stamp the suggestions —")
        print("     this set is the *independent* ground truth.")
        print("  6. When done: Project → Export → JSON.")
        print(f"  7. python notebooks/{Path(__file__).name} "
              f"--import-from <that_export.json>")
        return

    if args.import_from:
        path = Path(args.import_from)
        if not path.exists():
            raise SystemExit(f"missing {path}")
        items = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(items, list):
            raise SystemExit("expected a JSON list at the top of the LS export")
        verified = from_ls_export(items, annotator=args.annotator)

        if not BACKUP.exists():
            shutil.copy(INPUT, BACKUP)
            print(f"backed up Codex draft → {BACKUP.relative_to(ROOT)}")
        with INPUT.open("w", encoding="utf-8") as f:
            for r in verified:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {INPUT.relative_to(ROOT)}  "
              f"({len(verified)} verified rows)")

        # quick stats so you can sanity-check before trusting it as gold
        tc = Counter(s["type"] for r in verified for s in r["spans"])
        uncertain = sum(1 for r in verified if r["uncertain"])
        with_notes = sum(1 for r in verified if r["notes"])
        rows_with_spans = sum(1 for r in verified if r["spans"])
        print(f"  rows with ≥1 span: {rows_with_spans}/{len(verified)}")
        print(f"  spans by type:     {dict(tc.most_common())}")
        print(f"  uncertain flagged: {uncertain}")
        print(f"  rows with notes:   {with_notes}")
        return

    ap.print_help()


if __name__ == "__main__":
    main()
