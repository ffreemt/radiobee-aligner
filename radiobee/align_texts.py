"""Align texts based on aset, src_text, tgt_text."""
# pylint: disable=unused-variable

from typing import List, Tuple, Union
from logzero import logger


# fmt: off
def align_texts(
        aset: List[Tuple[Union[str, float], Union[str, float], Union[str, float]]],
        src_text: List[str],
        tgt_text: List[str],
) -> List[Tuple[Union[str], Union[str], Union[str, float]]]:
    # fmt: on
    """Align texts (paras/sents) based on aset, src_text, tgt_text.

    Args:
        aset: align set
        src_text: source text
        tgt_text: target text

    Returns:
        aligned texts with possible mertics
    """
    xset, yset, metrics = zip(*aset)  # unzip aset
    xset = [elm for elm in xset if elm != ""]
    yset = [elm for elm in yset if elm != ""]

    if (len(xset), len(yset)) != (len(tgt_text), len(src_text)):
        logger.warning(
            " (%s, %s) != (%s, %s) ", len(xset), len(yset), len(tgt_text), len(src_text)
        )
        # raise Exception(" See previous message")

    texts = []
    for elm in aset:
        elm0, elm1, elm2 = elm
        _ = []

        # src_text first
        if isinstance(elm1, str):
            _.append("")
        else:
            _.append(src_text[int(elm1)])

        if isinstance(elm0, str):
            _.append("")
        else:
            _.append(tgt_text[int(elm0)])

        if isinstance(elm2, str):
            _.append("")
        else:
            _.append(round(elm2, 2))

        texts.append(tuple(_))

    # return [("", "", 0.)]
    return texts
