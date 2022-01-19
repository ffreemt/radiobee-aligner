"""Convert paras to sents."""
# pylint: disable=unused-import, too-many-branches, ungrouped-imports

from typing import Callable, List, Optional, Tuple, Union

from itertools import zip_longest
import numpy as np
import pandas as pd
from logzero import logger

from radiobee.align_sents import align_sents
from radiobee.seg_text import seg_text
from radiobee.detect import detect

try:
    from radiobee.shuffle_sents import shuffle_sents
except Exception as exc:
    logger.error("shuffle_sents not available: %s, using align_sents", exc)
    shuffle_sents = lambda x1, x2, lang1="", lang2="": align_sents(x1, x2)  # noqa


def paras2sents(
    paras_: Union[pd.DataFrame, List[Tuple[str, str, Union[str, float]]], np.ndarray],
    align_func: Optional[Union[Callable, str]] = None,
    lang1: Optional[str] = None,
    lang2: Optional[str] = None,
) -> List[Tuple[str, str, Union[str, float]]]:
    """Convert paras to sents using align_func.

    Args:
        paras_: list of 3-tuples or numpy or pd.DataFrame
        lang1: fisrt lang code
        lang2: second lang code
        align_func: func used in the sent level
            if set to None, default to align_sents
    Returns:
        list of sents (possible with likelihood for shuffle_sents)
    """
    # wrap everything in pd.DataFrame
    # necessary to make pyright happy
    paras = pd.DataFrame(paras_).fillna("")

    # take the first three columns at maximum
    paras = paras.iloc[:, :3]

    if len(paras.columns) < 2:
        logger.error(
            "Need at least two columns, got %s",
            len(paras.columns)
        )
        raise Exception("wrong data")

    # append the third col (all "") if there are only two cols
    if len(paras.columns) < 3:
        paras.insert(2, "likelihood", [""] * len(paras))

    if lang1 is None:
        lang1 = detect(" ".join(paras.iloc[:, 0]))
    if lang2 is None:
        lang2 = detect(" ".join(paras.iloc[:, 1]))

    left, right = [], []
    row0, row1 = [], []
    for elm0, elm1, elm2 in paras.values:
        sents0 = seg_text(elm0, lang1)
        sents1 = seg_text(elm1, lang2)
        if isinstance(elm2, float) and elm2 > 0:
            if row0 or row1:
                left.append(row0)
                right.append(row1)
            row0, row1 = [], []  # collect and prepare

            if sents0:
                left.append(sents0)
            if sents1:
                right.append(sents1)
        else:
            if sents0:
                row0.extend(sents0)
            if sents1:
                row1.extend(sents1)
    # collect possible last batch
    if row0 or row1:
        left.append(row0)
        right.append(row1)

    # res = [*zip(left, right)]

    # align each batch using align_func

    # ready align_func
    if align_func is None:
        align_func = align_sents
    if isinstance(align_func, str) and align_func.startswith("shuffle") or not isinstance(align_func, str) and align_func.__name__ in ["shuffle_sents"]:
        align_func = lambda row0, row1: shuffle_sents(row0, row1, lang1=lang1, lang2=lang2)  # noqa
    else:
        align_func = align_sents

    res = []
    for row0, row1 in zip(left, right):
        try:
            _ = align_func(row0, row1)
        except Exception as exc:
            logger.error("errors: %s, resorting to zip_longest", exc)
            _ = [*zip_longest(row0, row1, fillvalue="")]

        # res.append(_)
        res.extend(_)

    return res
