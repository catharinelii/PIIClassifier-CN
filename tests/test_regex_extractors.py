"""Tests for Tier-0 regex extractors.

Coverage priorities:
1. The Chinese ID checksum is the most error-prone piece — test it hard.
2. Overlap resolution: ID-validated wins over generic-account on same span.
3. False-positive resistance: don't fire on case_no / order_num patterns.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from piiclassifier_cn.extractors.regex_extractors import (  # noqa: E402
    _ID_CHECK_MAP,
    _ID_WEIGHTS,
    _verify_id18,
    anonymize,
    extract_accounts,
    extract_addresses_rough,
    extract_all,
    extract_emails,
    extract_ids,
    extract_phones,
    extract_plates,
)
from piiclassifier_cn.spans import PIIType  # noqa: E402


def make_valid_id(prefix17: str) -> str:
    """Compute the spec-valid 18-digit ID for a given 17-digit prefix.

    Using this helper in tests means we can never typo a check digit — the
    fixture and the implementation share the same algorithm.
    """
    assert len(prefix17) == 17 and prefix17.isdigit()
    check = _ID_CHECK_MAP[sum(int(c) * w for c, w in zip(prefix17, _ID_WEIGHTS)) % 11]
    return prefix17 + check


# ---------------------------------------------------------------------------
# Chinese ID checksum
# ---------------------------------------------------------------------------
class TestIDChecksum:
    # All "valid" fixtures are computed from the algorithm itself via
    # ``make_valid_id`` — see top of file for why.
    VALID_PREFIXES = [
        "11010519491231002",
        "44010120000101123",
        "11010119900101001",
    ]
    INVALID = [
        "11010519491231002A",  # bad char in check position
        "110105194912310028",  # wrong check digit (valid one ends in 'X')
        "111111111111111111",  # constant digits, won't satisfy weighted sum
    ]

    def test_known_valid(self) -> None:
        for prefix in self.VALID_PREFIXES:
            s = make_valid_id(prefix)
            assert _verify_id18(s), f"expected {s} to validate"

    def test_known_invalid(self) -> None:
        for s in self.INVALID:
            assert not _verify_id18(s), f"expected {s} to fail"

    def test_extractor_with_checksum_filters_bad_ids(self) -> None:
        good = make_valid_id("11010519491231002")  # ends in 'X'
        text = f"好的 {good} 坏的 110105194912310028"
        spans = extract_ids(text, verify_checksum=True)
        assert len(spans) == 1
        assert spans[0].text == good

    def test_extractor_without_checksum_keeps_bad_ids(self) -> None:
        good = make_valid_id("11010519491231002")
        text = f"好的 {good} 坏的 110105194912310028"
        spans = extract_ids(text, verify_checksum=False)
        assert len(spans) == 2


# ---------------------------------------------------------------------------
# Phones
# ---------------------------------------------------------------------------
class TestPhones:
    def test_mobile_basic(self) -> None:
        spans = extract_phones("电话13812345678号")
        assert len(spans) == 1
        assert spans[0].text == "13812345678"
        assert spans[0].type == PIIType.PHONE

    def test_mobile_must_start_with_1_then_3to9(self) -> None:
        # 12345678901 is 11 digits but second digit is 2; should not match.
        assert extract_phones("12345678901") == []

    def test_mobile_no_partial_match_inside_longer_digit_run(self) -> None:
        # A 12-digit string shouldn't be parsed as a phone (no boundaries).
        assert extract_phones("138123456789") == []

    def test_multiple_phones(self) -> None:
        text = "联系13800138000或13911112222"
        spans = extract_phones(text)
        assert {s.text for s in spans} == {"13800138000", "13911112222"}


# ---------------------------------------------------------------------------
# Plates
# ---------------------------------------------------------------------------
class TestPlates:
    def test_standard_plate(self) -> None:
        spans = extract_plates("车牌京A12345违章")
        assert [s.text for s in spans] == ["京A12345"]

    def test_new_energy_plate(self) -> None:
        spans = extract_plates("新能源车京AD12345")
        assert [s.text for s in spans] == ["京AD12345"]


# ---------------------------------------------------------------------------
# Emails
# ---------------------------------------------------------------------------
class TestEmails:
    def test_basic(self) -> None:
        spans = extract_emails("联系me@example.com谢谢")
        assert [s.text for s in spans] == ["me@example.com"]


# ---------------------------------------------------------------------------
# Address (rough)
# ---------------------------------------------------------------------------
class TestAddressRough:
    def test_basic_village(self) -> None:
        spans = extract_addresses_rough("市民住在大兴区青云店镇沙堆营村反映")
        assert len(spans) >= 1
        # Should at least include the district-and-village core.
        assert any("大兴区" in s.text and "村" in s.text for s in spans)

    def test_compound_address(self) -> None:
        spans = extract_addresses_rough("地址：大兴区高庄村12号")
        assert any("大兴区" in s.text for s in spans)


# ---------------------------------------------------------------------------
# extract_all + overlap resolution
# ---------------------------------------------------------------------------
class TestExtractAll:
    def test_id_wins_over_account_on_same_chars(self) -> None:
        """A checksum-valid 18-digit ID is also matched by the 16-19 account
        regex. Overlap resolution must keep the ID and drop the account."""
        good = make_valid_id("11010519491231002")
        text = f"身份证 {good} 完毕"
        spans = extract_all(text)
        kinds = {s.type for s in spans}
        assert PIIType.ID in kinds
        assert PIIType.ACCOUNT not in kinds

    def test_no_double_count_on_phone(self) -> None:
        text = "电话13812345678"
        spans = extract_all(text)
        assert sum(1 for s in spans if s.type == PIIType.PHONE) == 1

    def test_real_complaint_sample(self) -> None:
        """A realistic Chinese complaint sentence touching multiple PII types."""
        good = make_valid_id("11010519491231002")
        text = (
            "市民反映，自己2022年8月在大兴区高庄村干外墙油漆，"
            f"拖欠人姓名张新军，身份证{good}，电话13800138000。"
        )
        spans = extract_all(text)
        kinds = {s.type for s in spans}
        assert PIIType.ID in kinds
        assert PIIType.PHONE in kinds
        assert PIIType.ADDRESS in kinds


class TestAnonymize:
    def test_replaces_with_placeholder_tokens(self) -> None:
        good = make_valid_id("11010519491231002")
        text = f"电话13800138000和ID{good}"
        out = anonymize(text)
        assert "13800138000" not in out
        assert good not in out
        assert "[PHONE]" in out
        assert "[ID]" in out

    def test_offsets_stable_with_multiple_spans(self) -> None:
        text = "A 13800138000 B 110105194912310029 C"
        out = anonymize(text)
        # Keep the non-PII text as-is.
        assert out.startswith("A [PHONE] B ")
        assert out.endswith(" C")


def _run() -> None:
    """Tiny test runner so we don't need pytest installed to verify."""
    import inspect
    total = passed = 0
    for cls_name, cls in list(globals().items()):
        if not (cls_name.startswith("Test") and inspect.isclass(cls)):
            continue
        inst = cls()
        for name, fn in inspect.getmembers(inst, predicate=inspect.ismethod):
            if not name.startswith("test_"):
                continue
            total += 1
            try:
                fn()
            except AssertionError as e:
                print(f"FAIL  {cls_name}.{name}: {e}")
            except Exception as e:
                print(f"ERROR {cls_name}.{name}: {type(e).__name__}: {e}")
            else:
                passed += 1
                print(f"ok    {cls_name}.{name}")
    print(f"\n{passed}/{total} passed")


if __name__ == "__main__":
    _run()
