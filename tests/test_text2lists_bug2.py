"""Test text2lists."""
from pathlib import Path
from radiobee.loadtext import loadtext
from radiobee.text2lists import text2lists


def test_text2lists_bug2():
    """Test text2lists data\问题2测试文件.txt."""
    filename = r"data\问题2测试文件.txt"
    text = loadtext(filename)  # noqa
    l1, l2 = text2lists(text)
    # assert l2[0] in [""]
    # assert "国际\n中\n双语" in l1[0]

    assert len(l1) == 4
    assert len(l2) == 5
