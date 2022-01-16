"""Test text2lists."""
from pathlib import Path
from radiobee.loadtext import loadtext
from radiobee.text2lists import text2lists


def test_text2lists_bug2():
    r"""Test text2lists data\问题2测试文件.txt."""
    filename = r"data\问题2测试文件.txt"
    textbug2 = loadtext(filename)  # noqa
    l1, l2 = text2lists(textbug2)

    assert len(l1) == 5
    assert len(l2) == 4
