"""Test align_sents."""
from radiobee.align_sents import align_sents


def test_align_sents():
    """Test align_sents."""
    lst1, lst2 = [
        "a",
        "bs",
    ], ["aaa", "34", "a", "b"]
    res = align_sents(lst1, lst2)

    assert res == [("a", "aaa"), ("a", "34"), ("bs", "a"), ("bs", "b")]
