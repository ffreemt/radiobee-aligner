"""Split text to sentences.

Use sentence_splitter if supported,
else use polyglot.text.Text

!apt install libicu-dev
!install pyicu pycld2 Morfessor
!pip install polyglot sentence_splitter
"""
# pylint: disable=

from typing import List, Optional, Union

import re
from tqdm.auto import tqdm
from polyglot.detect.base import logger as polyglot_logger
from polyglot.text import Detector, Text
from sentence_splitter import split_text_into_sentences

from logzero import logger

# turn of polyglot.text.Detector warning
polyglot_logger.setLevel("ERROR")


# fmt: off
# use sentence_splitter if supported
LANG_S = ["ca", "cs", "da", "nl", "en", "fi", "fr", "de",
          "el", "hu", "is", "it", "lv", "lt", "no", "pl",
          "pt", "ro", "ru", "sk", "sl", "es", "sv", "tr"]


def _seg_text(
        text: str,
        lang: Optional[str] = None,
        # qmode: bool = False,
        maxlines: int = 1000
) -> List[str]:
    # fmt: on
    """Split text to sentences.

    Use sentence_splitter if supported,
    else use polyglot.text.Text.sentences
    Blank lines will be removed.

    qmode: quick mode, skip split_text_into_sentences if True, default False
        vectors for all books are based on qmode=False.
        qmode=True is for quick test purpose only

    maxlines (default 1000), threshold for turn on tqdm progressbar
        set to <1 or a large number to turn it off
    """
    if lang is None:
        try:
            lang = Detector(text).language.code
        except Exception as exc:
            logger.info("text[:30]: %s", text[:30])
            logger.warning(
                "polyglot.text.Detector exc: %s, setting to 'en'",
                exc
            )
            lang = "en"

    # if not qmode and lang in LANG_S:
    if lang in LANG_S:
        _ = []
        lines = text.splitlines()
        # if maxlines > 1 and len(lines) > maxlines:
        if len(lines) > maxlines > 1:
            for para in tqdm(lines):
                if para.strip():
                    _.extend(split_text_into_sentences(para, lang))
        else:
            for para in lines:
                if para.strip():
                    _.extend(split_text_into_sentences(para, lang))
        return _

        # return split_text_into_sentences(text, lang)

    # empty "" text or blank to avoid Exception
    if not text.strip():
        return []

    return [elm.string for elm in Text(text, lang).sentences]


# fmt: off
def seg_text(
        lst: Union[str, List[str]],
        lang: Optional[str] = None,
        maxlines: int = 1000,
        extra: Optional[str] = None,
) -> List[str]:
    # fmt:on
    """Split a list of text.

    Arguments:
        lst: text or text list
        lang: optional lang code
        maxlines: (default 1000), threshold for turn on tqdm progressbar, set to <1 or a large number to turn it off
        extra: re.split(rf"{extra}, text) first
    Returns:
        list of splitted text.
    """
    if isinstance(lst, str):
        lst = [lst]

    if extra:
        # insert \n
        lst = [re.sub(rf"({extra})", r"\1\n", elm) for elm in lst]

    res = []
    for elm in lst:
        res.extend(_seg_text(
            elm,
            lang=lang,
            maxlines=maxlines,
        ))

    return res
