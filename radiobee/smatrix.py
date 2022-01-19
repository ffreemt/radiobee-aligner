"""Generate a similarity matrix (doc-term score matrix) based on textacy.representation.Vectorizer.

refer also to fast-scores fast_scores.py and gen_model.py (sklearn.feature_extraction.text.TfidfVectorizer).
originally docterm_scores.py.
"""
# pylint: disable=invalid-name, too-many-locals, too-many-arguments

from typing import Dict, Iterable, Optional, Union
from itertools import chain
import numpy as np
from psutil import virtual_memory
from more_itertools import ilen

from textacy.representations import Vectorizer

# from textacy.representations.vectorizers import Vectorizer
from logzero import logger

# from smatrix.gen_model import gen_model
from radiobee.gen_model import gen_model


# fmt: off
def smatrix(
        doc1: Iterable[Iterable[str]],  # List[List[str]],
        doc2: Iterable[Iterable[str]],
        model: Vectorizer = None,
        tf_type: str = 'linear',
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
    """Generate a doc-term score matrix based on textacy.representation.Vectorizer.

    Args
        doc1: tokenized doc of n1
        doc2: tokenized doc of n2
        model: if None, generate one ad hoc from doc1 and doc2 ("lucene-style tfidf").
        rest: refer to textacy.representation.Vectorizer
    Attributes
        vectorizer

    Returns
        n1 x n2 similarity matrix of float numbers
    """
    # make sure doc1/doc2 is of the right typing
    try:
        for xelm in iter(doc1):
            for elm in iter(xelm):
                assert isinstance(elm, str)
    except AssertionError as exc:
        raise AssertionError(" doc1 is not of the typing  Iterable[Iterable[str]] ") from exc
    except Exception as e:
        logger.error(e)
        raise
    try:
        for xelm in iter(doc2):
            for elm in iter(xelm):
                assert isinstance(elm, str)
    except AssertionError as exc:
        raise AssertionError(" doc2 is not of the typing  Iterable[Iterable[str]] ") from exc
    except Exception as e:
        logger.error(e)
        raise

    if model is None:
        model = gen_model(
            [*chain(doc1, doc2)],
            tf_type=tf_type,
            idf_type=idf_type,
            dl_type=dl_type,
            norm=norm,
            min_df=min_df,
            max_df=max_df,
            max_n_terms=max_n_terms,
            vocabulary_terms=vocabulary_terms
        )
        # docterm_scores.model = model
        smatrix.model = model

    # a1 = dt.toarray(), a2 = doc_term_matrix.toarray()
    # np.all(np.isclose(a1, a2))

    dt1 = model.transform(doc1)
    dt2 = model.transform(doc2)

    # virtual_memory().available / 8: 64bits float
    require_ram = ilen(iter(doc1)) * ilen(iter(doc2)) * 8
    if require_ram > virtual_memory().free:
        # logger.warning("virtual_memory().free: %s", virtual_memory().available)
        logger.warning("virtual_memory().free: %s", virtual_memory().free)
        logger.warning("memory required: %s", require_ram)

    if require_ram > virtual_memory().free * 10:
        logger.warning("You're likely to encounter memory problem, such as slowing down response and/or OOM.")

    # return dt1.doc(dt2.T)
    return dt2.toarray().dot(dt1.toarray().T)
