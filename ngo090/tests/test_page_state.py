"""The page a reload lands on.

Streamlit forgets the session when the tab reloads, so the visible page is
mirrored into the query string and read back on the first run. These tests
cover the translation both ways; the browser check is the real proof.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
import common  # noqa: E402


@pytest.mark.parametrize("mode", ("Live simulation", "Historical replay", "Transaction testing",
                                  "Cases", "Audit", "Approvals"))
def test_every_page_round_trips(mode):
    assert common.page_from_params(common.page_params(mode)) == (mode, None)


def test_a_case_page_carries_its_number():
    assert common.page_params("Case", 28) == {"page": "case", "case": "C-0028"}
    assert common.page_from_params({"page": "case", "case": "C-0028"}) == ("Case", 28)


def test_a_case_page_without_a_number_is_the_list():
    # A hand-typed or truncated URL has nothing to open: the list is one click away.
    assert common.page_from_params({"page": "case"}) == ("Cases", None)


def test_a_missing_or_unknown_page_falls_back_to_the_live_feed():
    assert common.page_from_params({}) == ("Live simulation", None)
    assert common.page_from_params({"page": "nonsense"}) == ("Live simulation", None)


def test_the_case_number_is_read_loosely():
    assert common.page_from_params({"page": "case", "case": "c-7"}) == ("Case", 7)


def test_the_list_never_carries_a_case_number():
    # Leaving a case must not leave its number in the URL behind.
    assert common.page_params("Cases", 28) == {"page": "cases"}
