"""Test paras2sents."""
# pylint: disable=invalid-name

import pandas as pd
from radiobee.paras2sents import paras2sents
from radiobee.shuffle_sents import shuffle_sents

file_loc = r"data/test-dual-zh-en.xlsx"
paras = pd.read_excel(file_loc, header=0)
paras = paras[["text1", "text2", "likelihood"]].fillna("")


def test_paras2sents_dual():
    """Test paras2sents_dual."""
    sents = paras2sents(paras)

    assert len(sents) > 202  # 208
    # assert not sents


def test_paras2sents_dual_model_s():
    """Test paras2sents_dual_model_s."""
    sents = paras2sents(paras, shuffle_sents)

    assert len(sents) > 201  # 207
    # assert not sents


_ = """
df = pd.DataFrame(
    [list(sent) + [""] if len(sent) == 2 else list(sent) for sent in sents]
).fillna("")

"""
