"""Detect language via polyglot and fastlid."""
# pylint: disable=

from typing import Any, Callable, List, Optional

from polyglot.text import Detector
import polyglot.detect.base
from polyglot.detect.base import UnknownLanguage
from fastlid import fastlid

from logzero import logger

polyglot.detect.base.logger.setLevel("ERROR")


def with_func_attrs(**attrs: Any) -> Callable:
    """Define func_attrs."""

    def with_attrs(fct: Callable) -> Callable:
        for key, val in attrs.items():
            setattr(fct, key, val)
        return fct

    return with_attrs


# @with_func_attrs(set_languages=None)
# def detect(text: str) -> str:
def detect(text: str, set_languages: Optional[List[str]] = None) -> str:
    """Detect language via polyglot and fastlid."""
    # if not text.strip(): return "en"
    try:
        _ = [(elm.code[:2], elm.confidence) for elm in Detector(text).languages]
        detect.lang_conf = _
        lang, conf = _[0]
    except UnknownLanguage:
        if set_languages is None:
            def_lang = "en"
        else:
            # def_lang = set_languages[-1]
            def_lang = set_languages[0]
        logger.warning(" UnknownLanguage exception: probably snippet too short, setting to %s", def_lang)
        lang, conf = def_lang, 0
    except Exception as exc:
        logger.error(exc)
        lang, conf = "en", 0

    del conf

    # set_languages = detect.set_languages
    if set_languages is None:
        return lang

    # set_languages is set
    if not isinstance(set_languages, (list, tuple)):
        logger.warning("set_languages (%s) ought to be a list/tuple")

    if lang in set_languages:
        return lang

    # lang not in set_languages, use fastlid
    fastlid.set_languages = set_languages
    lang, _ = fastlid(text)

    return lang
