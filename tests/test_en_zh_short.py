"""Test loadtext."""
# pylint: diable=invalid-name
import pytest

from fastlid import fastlid

from radiobee.loadtext import loadtext
from radiobee.files2df import files2df
from radiobee.file2text import file2text
from radiobee.lists2cmat import lists2cmat
from radiobee.cmat2tset import cmat2tset
from radiobee.gen_pset import gen_pset

en = loadtext("data/en.txt")
zh = loadtext("data/zh.txt")
testen = loadtext("data/testen.txt")
testzh = loadtext("data/testzh.txt")


def test_en_zh_short1():
    """Test en_zh_short."""
    lst1 = [elm for elm in en.splitlines() if elm.strip()]
    lst2 = [elm for elm in zh.splitlines() if elm.strip()]

    lang1, _ = fastlid(en)
    lang2, _ = fastlid(zh)

    cmat0 = lists2cmat(lst1, lst2)
    pset = gen_pset(cmat0)

    assert pset.__len__() > 2


def test_en_zh_short2():
    """Test en_zh_short testen testzh."""
    # en = testen.copy()
    # zh = testzh.copy()
    lst1a = [elm for elm in testen.splitlines() if elm.strip()]
    lst2a = [elm for elm in testzh.splitlines() if elm.strip()]

    lang1a, _ = fastlid(testen)
    lang2a, _ = fastlid(testzh)

    cmat1 = lists2cmat(lst1a, lst2a)
    pset = gen_pset(cmat1)

    assert pset.__len__() > 2


_ = """
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set()
sns.set_style("darkgrid")
cmap = "viridis_r"
plt.ion()

eps = 6
min_samples = 10


tset = pd.DataFrame(cmat2tset(cmat))
tset.columns = ["x", "y", "cos"]

df_ = tset

# """
