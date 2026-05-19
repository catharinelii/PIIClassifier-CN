"""Span types shared across extractors.

A Span is the universal currency of this pipeline: every extractor — regex,
neural NER, post-processor — produces Spans, and downstream products
(anonymizer, router-feature-builder, geo-analytics) consume them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable


class PIIType(str, Enum):
    """The PII categories we extract.

    Aligned with the OpenAI privacy-filter taxonomy so we can compare
    head-to-head, with two Chinese-specific additions (PLATE for license
    plates, ORG for company names that show up frequently in 投诉 text).
    """

    ADDRESS = "ADDRESS"
    PHONE = "PHONE"
    PERSON = "PERSON"
    EMAIL = "EMAIL"
    URL = "URL"
    DATE = "DATE"
    ACCOUNT = "ACCOUNT"  # bank, social-security, generic long-digit
    ID = "ID"            # Chinese 身份证, separate from generic accounts
    PLATE = "PLATE"      # 车牌, Chinese-specific
    ORG = "ORG"          # company / institution names
    SECRET = "SECRET"    # passwords, API keys (rare here)


@dataclass(frozen=True)
class Span:
    """A single PII span detected in a text.

    Attributes
    ----------
    start : int
        Character offset (inclusive) into the source text.
    end : int
        Character offset (exclusive) into the source text. ``text[start:end]``
        is the extracted surface form.
    text : str
        The literal surface form. Stored redundantly with start/end so
        downstream consumers don't have to keep the source around.
    type : PIIType
        Category of PII.
    confidence : float
        Extractor's confidence in [0, 1]. Regex with a passing checksum
        emits 1.0; fuzzy regex emits ~0.6–0.8; model output emits its
        decoded probability.
    source : str
        Which extractor produced it. Useful for ablations and debugging.
    """

    start: int
    end: int
    text: str
    type: PIIType
    confidence: float = 1.0
    source: str = "unknown"

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError(f"end ({self.end}) must be > start ({self.start})")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0,1], got {self.confidence}")

    def overlaps(self, other: "Span") -> bool:
        """True if this span shares any characters with ``other``."""
        return not (self.end <= other.start or other.end <= self.start)

    def length(self) -> int:
        return self.end - self.start


def resolve_overlaps(spans: Iterable[Span]) -> list[Span]:
    """Resolve overlapping spans by preferring (confidence, length).

    When two regex extractors fire on the same characters (e.g. a 19-digit
    account-number regex catches a string that's actually an 18-digit ID
    plus a stray digit), we want the higher-confidence — and as a tiebreak,
    longer — span to win.

    Returns spans sorted by ``start``.
    """
    sorted_spans = sorted(
        spans,
        key=lambda s: (-s.confidence, -s.length(), s.start),
    )
    kept: list[Span] = []
    for span in sorted_spans:
        if any(span.overlaps(k) for k in kept):
            continue
        kept.append(span)
    return sorted(kept, key=lambda s: s.start)
