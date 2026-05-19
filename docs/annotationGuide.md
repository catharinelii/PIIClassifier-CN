# PIIClassifier-CN — Annotation Guide
---

## 0. The 30-second rules

Before any rule about specific entity types:

1. **If you can't decide a span/type in 30 seconds, mark the *row* uncertain and move on.** Momentum matters more than any single label. We triage uncertains together at the end.
2. **When in doubt, prefer the simpler interpretation** — fewer spans, tighter boundaries. Over-labeling is harder to detect later than under-labeling.
3. **The Tier-0 pre-fill is a hint, not an oracle.** Accept what's right, fix what's wrong, add what's missing.
4. **Every rule in this guide can be wrong.** If you encounter an edge case the guide doesn't cover, write it into §6 (Edge Cases) *before* deciding — so future-you and your past-self stay aligned.

---

## 1. Label schema

Eight types we actively annotate. The list is closed for the gold test set — adding a new type means we re-do work.

| Type | One-line definition |
|---|---|
| `ADDRESS` | Specific geographic location of a person, building, or complaint subject. |
| `PHONE` | A personal contact number (mobile or landline). |
| `ID` | A Chinese national ID number (身份证), 15 or 18 digits. |
| `PERSON` | A specific human named in the text. |
| `ORG` | A specific named organization, company, or institution. (excluding gov bodies)|
| `DATE` | A specific calendar date or date-time. |
| `PLATE` | A vehicle license plate (车牌号). |
| `ACCOUNT` | A personal account/card number (bank, social security, etc.). |
| `EMAIL` | An email address. |
| `URL` | A web URL. |

We do **not** label: pronouns, generic roles ("老板", "村长"), policy references, public service numbers, relative time ("今天"), case numbers, order numbers.

---

## 2. Universal span rules

These apply to **every** entity type.

### 2.1 Boundaries

- Use **character offsets**, half-open: `[start, end)`. `text[start:end]` is the surface form.
- **Trim** leading/trailing whitespace and punctuation (`，。：；、（）"" '"`).
- **Trim** leading verbs and prepositions: 住在, 位于, 来自, 在.
  - "市民住在大兴区高庄村" → span is `大兴区高庄村`, not `住在大兴区高庄村`.
- **Trim** trailing connectives: 的, 等, 之类.
- **Do not trim** functional address suffixes (号, 楼, 单元, 室) — they're part of the address.

### 2.2 Overlap

- **One label per character.** If two types could apply, pick the more specific:
  - `ID` > `ACCOUNT` (a valid 身份证 is also a long digit string — choose ID)
  - `ORG` > `ADDRESS` when the surface form is the institution name itself ("大兴区社保中心" → ORG, even though "大兴区" is inside it)
- **Prefer the outer span** when one nests inside another (full address, not its sub-components).

### 2.3 Multiple instances

- The same entity mentioned twice = two separate spans. ("大兴区...大兴区..." in one paragraph → two `ADDRESS` spans.)
- A list of entities joined by 和 / 、 = one span per entity. ("张三和李四" → two `PERSON` spans.)

### 2.4 What never gets a span

- Case numbers: `兴[2024]-0335006`, `兴[2023]-0471093`
- Order numbers: `热线-231118-024255`, `网络-240729-048117`
- Pure numeric reference codes that aren't IDs/phones/accounts
- Pronouns: 我、自己、对方、本人、市民、群众
- Public-service phone numbers: `12345`, `110`, `119`, `120`, `10086`, `96110`

---

## 3. Per-type rules with examples

### 3.1 `ADDRESS`

**What counts.** A specific place: an administrative chain (province/city/district/township/village), a road + number, a building, a small business name with location context.

**Span scope.** Start at the leftmost place token (typically the highest administrative unit present). End at the deepest place token (unit/room/building name) before the next non-address token.

| Example | Span | Notes |
|---|---|---|
| 市民住在大兴区高庄村 | `大兴区高庄村` | Drop the verb. |
| 大兴区青云店镇沙堆营村 | `大兴区青云店镇沙堆营村` | Full chain. |
| 北京市大兴区金苑路23号 | `北京市大兴区金苑路23号` | Include city if present. |
| 大兴区高米店街道郁花园二里7号楼5单元 | `大兴区高米店街道郁花园二里7号楼5单元` | Include unit, no further sub-units. |
| 庞各庄西瓜小镇市场 | `庞各庄西瓜小镇` | Include named-place; exclude generic "市场" trailer. |

**What does NOT count as ADDRESS.**

- `大兴区` mentioned in policy context: "大兴区的规定" → no span (referring to the *jurisdiction*, not a location).
- "城市", "农村", "市区" — generic, not specific.
- "对面", "附近", "南面" — relative, not addresses on their own.
- Building names without geographic anchor: "国贸大厦" alone, when context doesn't make clear which one → annotator's judgment; lean toward no-label and mark row uncertain.

**Sticky edge case — institution + address.**

When an ORG name *contains* an address (e.g. `大兴区社保中心`), label the **whole thing as ORG**, not as ADDRESS. We accept losing the address signal here in exchange for clean type labels. If the address is repeated *elsewhere* in the row, label that occurrence as ADDRESS.

---

### 3.2 `PHONE`

**What counts.** Personal contact numbers — mobiles or landlines.

| Example | Span | Notes |
|---|---|---|
| 电话13800138000 | `13800138000` | Bare mobile. |
| 联系010-12345678 | `010-12345678` | Landline with area code; keep the dash. |
| 老板电话：18710106706 | `18710106706` | Drop the label "老板电话：". |
| 12345 | (none) | Public-service line, not PII. |
| 客服电话400-800-1234 | (none) | 400-number = business, public. Don't label. |

**Edge case.** If a row contains the same number twice (once as text, once after "电话："), label both occurrences.

---

### 3.3 `ID`

**What counts.** Chinese 身份证 — 18 digits (checksum-valid) or 15 digits.

| Example | Span | Notes |
|---|---|---|
| 身份证：110105194912310023 | `110105194912310023` *if checksum-valid* | If 18 digits but checksum fails → no span, mark row uncertain. |
| 身份证号：6226281977061737 | `622628197706173` | 15-digit ID; rare but valid. |
| 兴[2024]-0335006 | (none) | Case number, not an ID. |
| 91110000123456789X | (none) | This is a 统一社会信用代码 (corporate ID), 18 chars. We don't label it for now. |

**Hard rule:** an 18-digit candidate that fails the checksum gets no `ID` label. If the row otherwise still has clear PII, label that; if not, leave the row with whatever spans you have.

---

### 3.4 `PERSON`

**What counts.** Specific named humans. Typically 2–4 Chinese characters (family name + given name).

| Example | Span | Notes |
|---|---|---|
| 拖欠人姓名张新军 | `张新军` | Strip the role label. |
| 老板姓名：岳世民 | `岳世民` | Strip the label and colon. |
| 张三和李四 | `张三`, `李四` | Two separate spans. |
| 张先生 | `张` | Family name only; drop the honorific. |
| 小张, 老李 | `张`, `李` | Drop the informal prefix; family name remains. |
| 市民, 自己, 对方, 本人 | (none) | Pronoun-like, not PII. |
| 村长, 老板, 工人, 房东 | (none) | Role description, not PII. |
| 物业经理 | (none) | Role description. |
| 物业经理王大伟 | `王大伟` | Role + name → span only the name. |

**Sticky edge case — single-character names.** Rare in complaint text. If you see "刘" used alone to refer to a person ("刘说他..."), prefer no-label and mark the row uncertain.

---

### 3.5 `ORG`

**What counts.** Specific named organizations, companies, institutions.

| Example | Span | Notes |
|---|---|---|
| 嘉涂乐公司 | `嘉涂乐公司` | Include the 公司 suffix. |
| 大兴区社保中心 | `大兴区社保中心` | Whole institution name; do NOT also tag the address. |
| 北京市大兴区第一人民医院 | `北京市大兴区第一人民医院` | Include geographic prefix when it's part of the official name. |
| 世纪福超市（南高路） | `世纪福超市` | Drop the parenthetical location qualifier. |
| 一家公司 | (none) | Not specific. |
| 政府, 村委会 | (none) | Generic; not specific. |

**Sticky edge case — small businesses identified only by location.** "南高路超市" without a name → leave un-labeled; this is just a place reference and ADDRESS should already cover the road.

---

### 3.6 `DATE`

**What counts.** Specific calendar dates or date-times.

| Example | Span | Notes |
|---|---|---|
| 2023年10月28日 | `2023年10月28日` | Full date. |
| 2022年8月 | `2022年8月` | Year + month is specific enough. |
| 2023-10-28 08:40 | `2023-10-28 08:40` | Include time. |
| 2022年 | `2022年` | Year alone is specific. |
| 今天, 昨天, 上周 | (none) | Relative; not PII (and not stable in time). |
| 第二天 | (none) | Relative. |
| 早上, 晚上 | (none) | Time-of-day, no date. |

---

### 3.7 `PLATE`

**What counts.** Vehicle license plates. Province char + letter + 5–6 alphanumerics.

| Example | Span | Notes |
|---|---|---|
| 京A12345 | `京A12345` | Standard. |
| 京AD12345 | `京AD12345` | New-energy (6-char tail). |
| 鲁B·12345 | `鲁B·12345` | Include the dot if present. |

---

### 3.8 `ACCOUNT`, `EMAIL`, `URL`

Rare in this dataset (combined <2% of rows). Apply common sense; checksum on `ACCOUNT` is not feasible because card formats vary.

- `ACCOUNT`: long digit runs (16–19) clearly used as personal account numbers. Skip if it's a case number, order number, or shipping tracking number.
- `EMAIL`: standard `local@domain.tld` form. Include only the email itself.
- `URL`: full URL including scheme if present.

---

## 4. Row-level metadata

For each annotated row, in addition to spans:

| Field | Required? | Values |
|---|---|---|
| `uncertain` | yes (default false) | `true` if any decision in this row was a coin-flip. We'll triage these. |
| `notes` | no | Free-text comment on tricky decisions. Useful for guide revision. |
| `annotator` | yes | Your name/initials. |
| `annotated_at` | yes (auto) | ISO timestamp. |

**Don't** edit the `text` field. Don't fix typos in the source. We're labeling what's actually there.

---

## 5. Tier-0 pre-fill: how to use it

The annotation tool shows pre-filled spans from the Tier-0 regex extractors as **suggestions in a different color**.

- ✅ Accept → keep the span, type, boundaries as-is.
- ✏️ Edit → adjust boundaries or change type.
- ❌ Reject → delete the span.
- ➕ Add → draw a new span the regex missed.

**Vigilance points** — places where Tier-0 is known to misbehave:

1. Addresses with leading verbs ("市民住在大兴区..." regex includes "市民住在"). Trim.
2. Addresses that stop too early — regex stops at first place suffix. Extend to deepest unit if appropriate.
3. ACCOUNT spans that are actually case numbers. Reject.
4. 18-digit candidates that failed the checksum — these don't get pre-filled at all; you must add them as ID *only if* you can verify the checksum manually (the tool will tell you).

---

## 6. Edge case log (living section)

Add entries here as you encounter them. Format: situation, decision, rationale, date.

> *(empty until annotation starts — fill this section in as you go)*

---

## 7. Process

### 7.1 Workflow

1. Open the annotation tool. It picks the next un-annotated row from `data/to_annotate.jsonl`.
2. Read the row (10–20 seconds).
3. Confirm/edit/add spans (30–60 seconds typical).
4. Hit save+next. Tool appends to `data/gold.jsonl`.
5. Every 30 rows: take a 2-minute break. Annotation drift sets in fast.

### 7.2 Splits

After annotation is complete:

- **Test set (50 rows):** the cleanest 50 you trust. Locked. Used only for final evaluation. Never seen during training, tuning, or guide revision.
- **Dev set (~200 rows):** the rest. Used to tune Tier 1, debug, iterate.
- **Uncertain bucket:** triaged together at the end. May or may not enter the dev set depending on resolution.

### 7.3 Self-consistency check

After ~150 rows, re-annotate the first 20 rows blind (without seeing your earlier labels). Compute span-F1 against your past self. If <0.85, the guide needs tightening before continuing — don't paper over the inconsistency.

### 7.4 Sessions

- Cap any annotation session at 90 minutes. Quality drops sharply after.
- Commit to git at the end of each session: the JSONL file + any guide updates.
- Don't annotate when tired. Wrong labels are worse than fewer labels.

---

## 8. File format

`data/gold.jsonl` — one JSON object per line:

```json
{
  "id": "row_42",
  "text": "市民反映，自己住在大兴区高庄村，电话13800138000…",
  "spans": [
    {"start": 9, "end": 16, "type": "ADDRESS", "text": "大兴区高庄村"},
    {"start": 19, "end": 30, "type": "PHONE",  "text": "13800138000"}
  ],
  "uncertain": false,
  "notes": "",
  "annotator": "catharine",
  "annotated_at": "2026-05-14T10:30:00",
  "source_row_index": 12345
}
```

Hard rules on the file:

- One object per line. No trailing comma. No pretty-print.
- `spans` is sorted by `start`.
- `text[start:end]` must equal `text` for each span. The tool enforces this; if you ever edit the file by hand, preserve it.

---

## 9. Revision policy

This guide is **expected to change** in week 1 of annotation. Standard practice:

- Bumping a rule (clearer wording, new example) → just edit and commit. Log the change in §10.
- Adding a new edge case → §6, with date.
- Adding a new type → **stop annotating. Discuss.** Adding a type after the test set is built means re-annotating the test set.
- Renaming a type → never. If a name is wrong, it stays for the lifetime of the project.

---

