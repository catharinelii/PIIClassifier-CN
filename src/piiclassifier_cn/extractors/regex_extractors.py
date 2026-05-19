"""Tier-0 deterministic PII extractors for Chinese complaint text.

These are the cheap, near-perfect baseline. They handle entities with rigid
structure — phone numbers, national IDs (with a checksum), license plates,
emails, accounts — better than any neural model could, because the patterns
are *defined by spec*, not learned.

What we deliberately do NOT do here:
- Person-name extraction (regex is hopeless on Chinese names; that's Tier 1)
- Full address parsing (regex catches obvious cases; messy spans are Tier 1)
- Anything semantic ("does this address look like a complaint location vs a
  contextual reference?")

Public API:
    extract_phones(text) -> list[Span]
    extract_ids(text) -> list[Span]
    extract_plates(text) -> list[Span]
    extract_emails(text) -> list[Span]
    extract_accounts(text) -> list[Span]
    extract_addresses_rough(text) -> list[Span]
    extract_all(text) -> list[Span]   # combines + resolves overlaps
"""
from __future__ import annotations

import re
from typing import Iterable

from ..spans import PIIType, Span, resolve_overlaps

# ---------------------------------------------------------------------------
# Phones
# ---------------------------------------------------------------------------
# Chinese mobile: 11 digits starting with 1, second digit in [3-9].
# We require word boundaries so we don't gobble parts of longer digit strings
# (which are usually IDs or bank accounts).
_MOBILE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")

# Chinese landline: optional area code (3-4 digits, optionally parenthesised),
# optional dash, then 7-8 digit local number.
_LANDLINE_RE = re.compile(
    r"(?<!\d)(?:0\d{2,3}[-‐]?)?(?<![\d-])\d{7,8}(?!\d)"
)


def extract_phones(text: str) -> list[Span]:
    spans: list[Span] = []
    for m in _MOBILE_RE.finditer(text):
        spans.append(
            Span(
                start=m.start(),
                end=m.end(),
                text=m.group(),
                type=PIIType.PHONE,
                confidence=1.0,
                source="regex:mobile",
            )
        )
    # We deliberately skip the landline regex by default because it's noisy —
    # it would match every order_num and case_no fragment. Enable explicitly.
    return spans


# ---------------------------------------------------------------------------
# Chinese national ID (身份证) with checksum
# ---------------------------------------------------------------------------
# 18-digit ID: 6-digit administrative code + 8-digit DOB (YYYYMMDD) + 3-digit
# sequence + 1 check digit ('0'-'9' or 'X'/'x'). 15-digit IDs (pre-1999) are
# rarer but still appear; we handle both.
_ID_18_RE = re.compile(
    r"(?<!\d)"
    r"[1-9]\d{5}"                          # admin code, no leading zero
    r"(?:19|20)\d{2}"                      # year 1900-2099
    r"(?:0[1-9]|1[0-2])"                   # month
    r"(?:0[1-9]|[12]\d|3[01])"             # day
    r"\d{3}"                               # sequence
    r"[\dXx]"                              # check digit
    r"(?!\d)"
)
_ID_15_RE = re.compile(
    r"(?<!\d)"
    r"[1-9]\d{5}"                          # admin code
    r"\d{2}"                               # YY (19xx)
    r"(?:0[1-9]|1[0-2])"                   # month
    r"(?:0[1-9]|[12]\d|3[01])"             # day
    r"\d{3}"                               # sequence
    r"(?!\d)"
)

_ID_WEIGHTS = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
_ID_CHECK_MAP = "10X98765432"  # index 0 → '1', 1 → '0', 2 → 'X', etc.


def _verify_id18(s: str) -> bool:
    """Return True if a candidate 18-character ID has a valid checksum.

    See: GB 11643-1999 "Citizen Identification Number". The 18th digit is
    determined by the first 17 via a weighted modular sum.
    """
    if len(s) != 18:
        return False
    if not s[:17].isdigit():
        return False
    expected = _ID_CHECK_MAP[sum(int(c) * w for c, w in zip(s[:17], _ID_WEIGHTS)) % 11]
    return s[17].upper() == expected


def extract_ids(text: str, *, verify_checksum: bool = True) -> list[Span]:
    """Extract Chinese national IDs.

    Parameters
    ----------
    verify_checksum : bool
        If True (default), only emit 18-digit IDs whose checksum validates.
        Drops false-positive rate from ~30% to <1%.
    """
    spans: list[Span] = []
    for m in _ID_18_RE.finditer(text):
        s = m.group()
        if verify_checksum and not _verify_id18(s):
            continue
        spans.append(
            Span(
                start=m.start(),
                end=m.end(),
                text=s,
                type=PIIType.ID,
                confidence=1.0 if verify_checksum else 0.7,
                source="regex:id18+checksum" if verify_checksum else "regex:id18",
            )
        )
    # 15-digit IDs: emit at lower confidence (no checksum exists in the spec).
    for m in _ID_15_RE.finditer(text):
        spans.append(
            Span(
                start=m.start(),
                end=m.end(),
                text=m.group(),
                type=PIIType.ID,
                confidence=0.7,
                source="regex:id15",
            )
        )
    return spans


# ---------------------------------------------------------------------------
# License plates (机动车号牌)
# ---------------------------------------------------------------------------
# Standard plate: province char + letter + 5 alphanumerics (e.g. 京A12345).
# New-energy plates have 6 trailing alphanumerics (京AD12345). We accept both
# by allowing 5-6 trailing chars.
_PROVINCES = "京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼"
_PLATE_RE = re.compile(rf"[{_PROVINCES}][A-Z][A-HJ-NP-Z0-9]{{5,6}}")


def extract_plates(text: str) -> list[Span]:
    spans: list[Span] = []
    for m in _PLATE_RE.finditer(text):
        spans.append(
            Span(
                start=m.start(),
                end=m.end(),
                text=m.group(),
                type=PIIType.PLATE,
                confidence=0.95,
                source="regex:plate",
            )
        )
    return spans


# ---------------------------------------------------------------------------
# Emails
# ---------------------------------------------------------------------------
# NOTE: Python's `\w` is Unicode-aware by default and would match Chinese
# characters, so "联系me@example.com" would greedily include "联系". Restrict
# both localpart and domain to ASCII alnum + permitted punctuation.
_EMAIL_RE = re.compile(r"[A-Za-z0-9._+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9.-]+")


def extract_emails(text: str) -> list[Span]:
    return [
        Span(
            start=m.start(),
            end=m.end(),
            text=m.group(),
            type=PIIType.EMAIL,
            confidence=0.95,
            source="regex:email",
        )
        for m in _EMAIL_RE.finditer(text)
    ]


# ---------------------------------------------------------------------------
# Generic long-digit accounts (bank cards, social-security, etc.)
# ---------------------------------------------------------------------------
# We catch unbroken digit runs of 16-19 chars that are NOT already classified
# as IDs (handled by overlap resolution downstream). Confidence is moderate
# because these patterns overlap a lot with case numbers.
_ACCOUNT_RE = re.compile(r"(?<!\d)\d{16,19}(?!\d)")


def extract_accounts(text: str) -> list[Span]:
    return [
        Span(
            start=m.start(),
            end=m.end(),
            text=m.group(),
            type=PIIType.ACCOUNT,
            confidence=0.5,
            source="regex:account-digits",
        )
        for m in _ACCOUNT_RE.finditer(text)
    ]


# ---------------------------------------------------------------------------
# Rough Chinese addresses
# ---------------------------------------------------------------------------
# Best-effort. Tier 1 (the neural NER) will do the heavy lifting here. The
# regex pattern is: a *district* head (X区/X县/X市) followed by up to ~30
# characters that end in a recognizable place suffix.
#
# We accept these as low-confidence (0.6) precisely so that when the neural
# tagger disagrees, the neural one wins.
# Greedy on the filler so we capture the LONGEST valid address (down to
# the deepest place suffix in the chain), e.g. "大兴区青云店镇沙堆营村"
# rather than stopping at "大兴区青云店镇".
_ADDR_RE = re.compile(
    r"[一-龥]{2,8}(?:区|县)"                       # 大兴区, 顺义县
    r"[一-龥A-Za-z0-9\-]{0,30}"                    # filler (greedy)
    r"(?:村|镇|乡|街道|社区|小区|路|街|号楼?|大厦|公寓|花园|院)"
)


def extract_addresses_rough(text: str) -> list[Span]:
    return [
        Span(
            start=m.start(),
            end=m.end(),
            text=m.group(),
            type=PIIType.ADDRESS,
            confidence=0.6,
            source="regex:address-rough",
        )
        for m in _ADDR_RE.finditer(text)
    ]


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------
def extract_all(text: str) -> list[Span]:
    """Run every Tier-0 extractor, resolve overlaps, return sorted spans.

    Overlap resolution prefers higher confidence, then longer span. So a
    checksum-validated ID (confidence 1.0) beats a generic account-number
    match (confidence 0.5) on the same characters.
    """
    if not text:
        return []
    raw: list[Span] = []
    raw.extend(extract_phones(text))
    raw.extend(extract_ids(text))
    raw.extend(extract_plates(text))
    raw.extend(extract_emails(text))
    raw.extend(extract_accounts(text))
    raw.extend(extract_addresses_rough(text))
    return resolve_overlaps(raw)


def anonymize(text: str, spans: Iterable[Span] | None = None) -> str:
    """Replace each PII span with a typed placeholder, in-place by offset.

    If ``spans`` is None, runs :func:`extract_all` first.

    Example:
        >>> anonymize("电话13800138000")
        '电话[PHONE]'
    """
    if spans is None:
        spans = extract_all(text)
    # Sort descending so replacements don't shift earlier offsets.
    parts = list(text)
    for s in sorted(spans, key=lambda s: s.start, reverse=True):
        parts[s.start : s.end] = list(f"[{s.type.value}]")
    return "".join(parts)
