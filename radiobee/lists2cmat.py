"""Convert two lists of str (texts) to correlation matrix."""
# pylint: disable=too-many-arguments, too-many-locals, unused-import

from typing import Dict, Iterable, List, Optional, Union  # noqa

import numpy as np
from textacy.representations import Vectorizer
from fastlid import fastlid

from radiobee.en2zh_tokens import en2zh_tokens
from radiobee.insert_spaces import insert_spaces
from radiobee.gen_model import gen_model
from radiobee.smatrix import smatrix


# fmt: off
def lists2cmat(
        text1: Union[str, Iterable[str]],
        text2: Union[str, Iterable[str]],
        # text1: Union[str, List[str]],
        # text2: Union[str, List[str]],
        lang1: Optional[str] = None,
        lang2: Optional[str] = None,
        model: Vectorizer = None,
        tf_type: str = "linear",
        idf_type: Optional[str] = "smooth",
        # dl_type: Optional[str] = "sqrt",  # "lucene-style tfidf"
        dl_type: Optional[str] = None,  #
        norm: Optional[str] = "l2",  # + "l2"
        min_df: Union[int, float] = 1,
        max_df: Union[int, float] = 1.0,
        max_n_terms: Optional[int] = None,
        vocabulary_terms: Optional[Union[Dict[str, int], Iterable[str]]] = None
) -> np.ndarray:
    # fmt: on
    """Convert two lists to cmat.

    Args:
        text1: refer smatrix
        text2: refer smatrix
        lang1: optional 1st lang code
        lang2: optional 2nd lang code
        dl_type: doc lenth
        idf_type: idf tyoe
        max_df: max doc freq
        max_n_terms: max n terms
        min_df: min doc freq
        model: optional model
        norm: norm
        tf_type: term freq type
        vocabulary_terms: vocab refer smatrix

    Returs
        cmat
    """
    if isinstance(text1, str):
        text1 = [text1]
    if isinstance(text2, str):
        text2 = [text2]

    set_languages = fastlid.set_languages
    fastlid.set_languages = ["en", "zh"]
    if lang1 is None:
        lang1, _ = fastlid(" ".join(text1))
    if lang2 is None:
        lang2, _ = fastlid(" ".join(text2))

    # restore fastlid.set_languages
    fastlid.set_languages = set_languages

    # en2zh_tokens
    def zh_tokens(textzh):
        return [insert_spaces(elm).split() for elm in textzh]

    if lang1 in ["zh"] and lang2 in ["en"]:
        vec1 = zh_tokens(text1)
        vec2 = en2zh_tokens(text2)
    elif lang1 in ["zh"] and lang2 in ["zh"]:
        vec1 = zh_tokens(text1)
        vec2 = zh_tokens(text2)
    elif lang1 in ["en"] and lang2 in ["en"]:
        vec1 = en2zh_tokens(text1)
        vec2 = en2zh_tokens(text2)

    # if lang1 in ["en"] and lang2 in ["zh"]:
    else:
        vec1 = en2zh_tokens(text1)
        vec2 = zh_tokens(text2)

    if model is None:
        model = gen_model(vec1)

    cmat = smatrix(
        vec1,
        vec2,
        model=model,
        tf_type=tf_type,
        idf_type=idf_type,
        dl_type=dl_type,
        norm=norm,
        min_df=min_df,
        max_df=max_df,
        max_n_terms=max_n_terms,
        vocabulary_terms=vocabulary_terms,
    )

    return np.array(cmat)
