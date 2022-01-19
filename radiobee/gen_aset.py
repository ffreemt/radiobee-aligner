"""Genereat align set (aset) based on pset (pair set), src_lang and tgt_len."""
# pylint: disable=unused-variable

from typing import List, Tuple, Union
from itertools import zip_longest

# from logzero import logger


# fmt: off
def gen_aset(
        pset: List[Tuple[Union[str, float], Union[str, float], Union[str, float]]],
        src_len: int,  # n_rows
        tgt_len: int,  # n_cols
) -> List[Tuple[Union[str, float], Union[str, float], Union[str, float]]]:
    # fmt: on
    """Genereat align set (aset) based on pset, src_lang and tgt_len.

    src_len, tgt_len = cmat.shape
    zip_longest(..., fillvalue="")

    Args:
        pset: [x(lang2 zh), y(lang1 en), cos]
        src_len: lang1 (en)
        tgt_len: lang2 (zh)

    Returns:
        aset:
        [0...tgt_len, 0...src_len]
        [0, 0, .]
        ...
        [tgt_len-1, src_len-1, .]
    """
    # empty pset []
    if not pset:
        return [*zip_longest(range(tgt_len), range(src_len), fillvalue="")]
    # empty [[]]
    if len(pset) == 1:
        if not pset[0]:
            return [*zip_longest(range(tgt_len), range(src_len), fillvalue="")]

    buff = []
    pos0, pos1 = -1, -1
    for elm in pset:
        # elm0, elm1, elm2 = elm
        elm0, elm1, *elm2 = elm
        elm0 = int(elm0)
        elm1 = int(elm1)
        interval = max(elm0 - pos0 - 1, elm1 - pos1 - 1)
        _ = zip_longest(range(pos0 + 1, elm0), range(pos1 + 1, elm1), [""] * interval, fillvalue="")
        buff.extend(_)
        buff.append(elm)
        pos0, pos1 = elm0, elm1

    # last batch if any
    elm0, elm1 = tgt_len, src_len
    interval = max(elm0 - pos0 - 1, elm1 - pos1 - 1)
    _ = zip_longest(range(pos0 + 1, elm0), range(pos1 + 1, elm1), [""] * interval, fillvalue="")
    buff.extend(_)

    return buff
