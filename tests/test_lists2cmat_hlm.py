"""Test lists2cmat."""
# pylint: disable=invalid-name

from itertools import zip_longest
from fastlid import fastlid
from radiobee.loadtext import loadtext
from radiobee.lists2cmat import lists2cmat

file1 = "data/test_en.txt"
file2 = "data/test_zh.txt"
file1 = "data/hlm-ch1-en.txt"
file2 = "data/hlm-ch1-zh.txt"

# assume English or Chinese
fastlid.set_languages = ["en", "zh", ]

text1 = loadtext(file1)
text2 = loadtext(file2)

lang1, _ = fastlid(text1)
lang2, _ = fastlid(text2)


def test_lists2cmat_hlm():
    """Test lists2cmat."""

    lst1, lst2 = [], []

    if text1:
        lst1 = [_.strip() for _ in text1.splitlines() if _.strip()]
    if text2:
        lst2 = [_.strip() for _ in text2.splitlines() if _.strip()]

    # en                zh
    len(lst1) == 135, len(lst2) == 55

    # cmat = texts2cmat(lst1, lst2, lang1, lang2)
    cmat = lists2cmat(lst1, lst2, lang1, lang2)

    assert cmat.shape == (36, 33)

    cmat21 = lists2cmat(lst2, lst1, lang2, lang1)

    assert cmat21.shape == (33, 36)
    assert lists2cmat(lst2, lst1).mean() > 0.05  # 0.09
