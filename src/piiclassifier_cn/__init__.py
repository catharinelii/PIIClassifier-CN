"""PIIClassifier-CN: a tiered PII extraction pipeline for Chinese complaint text."""
from .spans import PIIType, Span, resolve_overlaps

__all__ = ["PIIType", "Span", "resolve_overlaps"]
__version__ = "0.0.1"
