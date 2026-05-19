# Annotation Setup — Label Studio

This doc gets you from "nothing installed" to "I'm annotating my first row."
Estimated time: ~15 min the first time, ~1 min for every subsequent session.

## Why Label Studio (vs custom Streamlit)

Decided 2026-05-15. We picked Label Studio off-the-shelf because:
- We want to spend learning time on the NER model, not the annotation tool.
- LS's UX is exactly what we sketched (highlight-to-tag, uncertain flag,
  hotkeys, pre-fills) — built and battle-tested.
- Native JSON import/export round-trips with our `Span` schema via the
  `piiclassifier_cn.io.labelstudio` adapter.
- Inter-annotator agreement support is built-in (useful if we ever want
  multi-pass consistency checks).

## Prerequisites

- `data/to_annotate.jsonl` already produced (run `02_sample_for_annotation.py`).
- Python 3.10+ and pip available.
- ~1 GB free disk (Label Studio brings a lot of deps).
- Modern browser (Label Studio is web-based; runs at `localhost:8080`).

## 1. Install Label Studio

```bash
pip install label-studio
```

Takes a few minutes the first time (it pulls Django + a lot of ML deps).

## 2. Convert our data to LS import format

```bash
cd PIIClassifier-CN
python notebooks/03_export_to_label_studio.py
```

This produces:
- `data/to_annotate.label_studio.json` — the 300 tasks, each with Tier-0
  regex pre-fills as `predictions`.
- `docs/label_studio_config.xml` — the labeling-interface config XML
  (label types, hotkeys, uncertain checkbox, notes box).

## 3. Launch Label Studio

```bash
label-studio start
```

First-launch ritual:
1. Browser opens at `localhost:8080`.
2. Sign up (it's a local-only account, just makes a SQLite user).
3. Create a new project. Name it whatever (e.g. "PIIClassifier-CN gold").

## 4. Wire up the project

In your new project's **Settings → Labeling Interface**:
1. Click **Code** (top-right toggle), not the visual editor.
2. Paste the contents of `docs/label_studio_config.xml`.
3. Click **Save**.

In **Settings → Cloud Storage** (optional, skip for now): we run local.

## 5. Import the tasks

In the project view:
1. Click **Import** → **Upload Files**.
2. Drop `data/to_annotate.label_studio.json`.
3. After upload, LS shows 300 tasks. The ones with Tier-0 pre-fills will
   appear with the predictions panel highlighted.

## 6. Start annotating

Click any task → the annotation UI appears.

### Keyboard shortcuts

| Key | What it does |
|---|---|
| `a` | Tag selection as `ADDRESS` |
| `p` | Tag selection as `PHONE` |
| `i` | Tag selection as `ID` |
| `n` | Tag selection as `PERSON` (name) |
| `o` | Tag selection as `ORG` |
| `d` | Tag selection as `DATE` |
| `l` | Tag selection as `PLATE` (license) |
| `c` | Tag selection as `ACCOUNT` (card) |
| `e` | Tag selection as `EMAIL` |
| `u` | Tag selection as `URL` |
| `s` | Tag selection as `SECRET` |
| `Ctrl+Enter` | Submit + load next task |
| `Esc` | Clear current selection |

To **delete a Tier-0 pre-fill you disagree with**: click on the colored
span → press `Backspace`. The "predictions" pre-fills don't enter the
final annotation until you confirm them by submitting — but it's still
visually noisy if you leave wrong ones around, so delete as you go.

To **flag a row as uncertain**: check the "Uncertain — flag for review"
box at the bottom before submitting. The row still gets saved; it's
just flagged for later triage. Add a note in the textarea explaining why.

### Workflow recommendations

- Cap each session at **~90 minutes**. Quality drops sharply after.
- Take a 2-minute break every **30 rows**.
- The **first ~50 rows** are the highest-learning ones — you'll discover
  edge cases that need entries in `docs/annotationGuide.md §6`. Add them
  as you encounter them.
- Annotate the test set portion (first 50 rows, or however we slice it
  later) with **extra care** — these are the ground truth that every
  model gets evaluated against.

## 7. Export annotations back to our schema

Once you've annotated (some or all of) the 300 rows:

In Label Studio:
1. Project view → **Export**.
2. Format: **JSON** (the default, full export — not "min" or "CSV").
3. Save the file to `data/label_studio_export.json`.

Then:
```bash
python notebooks/04_import_from_label_studio.py
```

This produces `data/gold.jsonl` — our canonical gold dataset, ready for
training/evaluation. You can re-export and re-import as many times as
you want; each run overwrites `gold.jsonl`.

## 8. Stopping and resuming

- Label Studio state lives in `~/.local/share/label-studio` (or your
  platform's equivalent). Re-running `label-studio start` resumes where
  you left off.
- Annotations are auto-saved to a draft as you make them; submit makes
  them final.
- To reset everything: `rm -rf ~/.local/share/label-studio` (don't run
  this once you've done real work — there's no undo).

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| "Cannot connect to port 8080" | Another process is on 8080. Run `label-studio start --port 8090` instead. |
| Chinese chars render as ▯ | Browser font issue, not LS. Try a different browser. |
| Pre-fills don't appear | The task's `predictions` block may be malformed — re-run `03_export_to_label_studio.py`. |
| Hotkeys don't work | Click into the text area first; LS only fires hotkeys when focus is in the labeling pane. |
| Export is empty | LS only exports *submitted* annotations by default — drafts aren't included. Check that you submitted each task. |
| "Label X not found" on import | Schema drift — the export references a label not in our current config. The `from_label_studio_export()` function tolerates this (skips unknown labels). |

## Files this workflow touches

```
PIIClassifier-CN/
  data/
    to_annotate.jsonl              # in:  300 rows we want labeled
    to_annotate.label_studio.json  # out: LS import file (generated)
    label_studio_export.json       # in:  LS export file (you save it here)
    gold.jsonl                     # out: our final labeled gold (generated)
  docs/
    annotationGuide.md             # the rulebook — read first
    annotationSetup.md             # this file
    label_studio_config.xml        # the LS labeling-interface XML
  notebooks/
    03_export_to_label_studio.py   # to_annotate.jsonl → LS import format
    04_import_from_label_studio.py # LS export → gold.jsonl
  src/piiclassifier_cn/io/
    labelstudio.py                 # the adapter library
  tests/
    test_labelstudio_io.py         # round-trip tests
```
