"""Separate text to zh en lists."""
# pylint: disable=

from typing import Iterable, List, Tuple, Union  # noqa
from fastlid import fastlid
from logzero import logger


def text2lists(text: Union[Iterable[str], str]) -> List[Tuple[str, str]]:
    """Separate text to zh en lists."""
    if not isinstance(text, str) and isinstance(text, Iterable):
        try:
            text = "\n".join(text)
        except Exception as e:
            logger.error(e)
            raise

    set_languages = ["en", "zh"]
    fastlid.set_languages = set_languages
    list1 = []
    list2 = []  # for determining en-zh or zh-en
    lang0, _ = fastlid(text[:15000])
    res = ""
    left = False  # start with left list1

    for elm in [_ for _ in text.splitlines() if _.strip()]:
        lang, _ = fastlid(elm)
        if lang == lang0:
            res = res + "\n" + elm
        else:
            left = not left
            if left:
                list1.append(res.strip())
            else:
                list2.append(res.strip())  # strip first \n
            res = elm
            lang0 = lang

    # find offset

    left = []  # noqa
    right = []  # noqa

    return [("", "")]
