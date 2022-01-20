"""Test paras2sents."""
# pylint: disable=invalid-name

import numpy as np
import pandas as pd
from radiobee.paras2sents import paras2sents
from radiobee.shuffle_sents import shuffle_sents

file_loc = r"data/test-dual-zh-en.xlsx"
paras = pd.read_excel(file_loc, header=0)
paras = paras[["text1", "text2", "likelihood"]].fillna("")


def test_paras2sents_dual():
    """Test paras2sents_dual."""
    sents = paras2sents(paras)

    assert np.array(sents).shape.__len__() > 1

    assert len(sents) > 202  # 208
    # assert not sents


def test_paras2sents_dual_model_s():
    """Test paras2sents_dual_model_s."""
    sents1 = paras2sents(paras, shuffle_sents)

    # assert np.array(sents1).shape.__len__() > 1
    assert pd.DataFrame(sents1).shape.__len__() > 1

    assert len(sents1) > 201  # 207
    # assert not sents


_ = """
df = pd.DataFrame(
    [list(sent) + [""] if len(sent) == 2 else list(sent) for sent in sents]
).fillna("")

"""
