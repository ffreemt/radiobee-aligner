"""Shuffle sents."""
# pylint: disable=

from typing import List, Optional, Tuple, Union

from fastlid import fastlid
from logzero import logger  # noqa

from radiobee.lists2cmat import lists2cmat
from radiobee.gen_pset import gen_pset
from radiobee.gen_aset import gen_aset
from radiobee.align_texts import align_texts


# fmt: off
def shuffle_sents(
        lst1: List[str],
        lst2: List[str],
        eps: float = 6,
        min_samples: int = 4,
        tf_type: str = "linear",
        idf_type: Optional[str] = None,
        dl_type: Optional[str] = None,
        norm: Optional[str] = None,
        lang1: Optional[str] = None,
        lang2: Optional[str] = None,
) -> List[Tuple[str, str, Union[str, float]]]:
    # fmt: on
    """shuffle sents to the right positions.

    Based on __main__.py.
    """
    set_languages = fastlid.set_languages
    fastlid.set_languages = ["en", "zh"]
    if lang1 is None:
        lang1, _ = fastlid(" ".join(lst1))
    if lang2 is None:
        lang2, _ = fastlid(" ".join(lst2))

    # restore fastlid.set_languages
    fastlid.set_languages = set_languages

    cmat = lists2cmat(
        lst1,
        lst2,
        tf_type=tf_type,
        idf_type=idf_type,
        dl_type=dl_type,
        norm=norm,
        lang1=lang1,
        lang2=lang2,
    )

    pset = gen_pset(
        cmat,
        eps=eps,
        min_samples=min_samples,
        delta=7,
    )

    src_len, tgt_len = cmat.shape
    aset = gen_aset(pset, src_len, tgt_len)

    final_list = align_texts(aset, lst2, lst1)

    return final_list

    # return [("", "")]
