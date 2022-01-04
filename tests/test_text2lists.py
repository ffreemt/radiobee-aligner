"""Test text2lists."""
from radiobee.text2lists import text2lists
from radiobee.loadtext import loadtext


def test_text2lists():
    """Test text2lists data\test-dual.txt."""
    filename = r"data\test-dual.txt"
    text = loadtext(filename)
