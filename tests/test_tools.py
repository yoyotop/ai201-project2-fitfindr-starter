import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from tools import search_listings, create_fit_card
def test_search_returns_results():
    results = search_listings(
        "vintage graphic tee",
        None,
        50
    )
    assert len(results) > 0

def test_search_empty_results():
    results = search_listings(
        "designer ballgown",
        "XXS",
        5
    )
    assert results == []

def test_fit_card_empty_outfit():
    result = create_fit_card("", {})
    assert "missing" in result.lower()