"""Shuffle sents."""
# pylint: disable=unused-import, too-many-arguments, too-many-locals,

from typing import List, Optional, Tuple, Union

import pandas as pd
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
    """Shuffle sents to the right positions.

    Based on __main__.py.

    eps: float = 6
    min_samples: int = 4
    tf_type: str = "linear"
    idf_type: Optional[str] = None
    dl_type: Optional[str] = None
    norm: Optional[str] = None
    lang1: Optional[str] = "en"
    lang2: Optional[str] = "zh"
    """
    set_languages = fastlid.set_languages
    # fastlid.set_languages = ["en", "zh"]
    fastlid.set_languages = None

    if lang1 is None:
        lang1, _ = fastlid(" ".join(lst1))
    if lang2 is None:
        lang2, _ = fastlid(" ".join(lst2))

    # restore fastlid.set_languages
    fastlid.set_languages = set_languages

    lang_dicts = ["en", "zh"]
    if lang1 in lang_dicts and lang2 in lang_dicts:
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
    else:  # use model_s
        from radiobee.model_s import model_s  # pylint: disable=import-outside-toplevel
        vec1 = model_s.encode(lst1)
        vec2 = model_s.encode(lst2)
        # cmat = vec1.dot(vec2.T)
        cmat = vec2.dot(vec1.T)

    shuffle_sents.cmat = cmat
    shuffle_sents.lang1 = lang1
    shuffle_sents.lang2 = lang2

    pset = gen_pset(
        cmat,
        eps=eps,
        min_samples=min_samples,
        delta=7,
    )

    src_len, tgt_len = cmat.shape
    aset = gen_aset(pset, src_len, tgt_len)

    final_list = align_texts(aset, lst2, lst1)

    # return final_list

    # swap columns 0, 1
    _ = pd.DataFrame(final_list)

    _ = _.iloc[:, [1, 0] + [*range(2, _.shape[1])]]

    return _.to_numpy().tolist()
