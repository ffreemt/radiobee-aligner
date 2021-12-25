"""Generate a model (textacy.representations.Vectorizer).

vectorizer = Vectorizer(
    tf_type="linear", idf_type="smooth", norm="l2",
    min_df=3, max_df=0.95)
doc_term_matrix = vectorizer.fit_transform(tokenized_docs)
doc_term_matrix

tokenized_docs = [insert_spaces(elm).split() for elm in textzh]
"""
from typing import Dict, Iterable, List, Optional, Union  # noqa

from textacy.representations import Vectorizer
from logzero import logger


# fmt: off
def gen_model(
        tokenized_docs: Iterable[Iterable[str]],  # List[List[str]],
        tf_type: str = 'linear',
        idf_type: Optional[str] = "smooth",
        dl_type: str = None,  # Optional[str] = "sqrt" “lucene-style tfidf”
        norm: Optional[str] = "l2",  # + "l2"
        min_df: Union[int, float] = 1,
        max_df: Union[int, float] = 1.0,
        max_n_terms: Optional[int] = None,
        vocabulary_terms: Optional[Union[Dict[str, int], Iterable[str]]] = None
) -> Vectorizer:
    # fmt: on
    """Generate a model (textacy.representations.Vectorizer).

    Args:
        doc: tokenized docs

        (refer to textacy.representation.Vectorizer)
        tf_type: Type of term frequency (tf) to use for weights' local component:

            - "linear": tf (tfs are already linear, so left as-is)
            - "sqrt": tf => sqrt(tf)
            - "log": tf => log(tf) + 1
            - "binary": tf => 1

        idf_type: Type of inverse document frequency (idf) to use for weights'
            global component:

            - "standard": idf = log(n_docs / df) + 1.0
            - "smooth": idf = log(n_docs + 1 / df + 1) + 1.0, i.e. 1 is added
              to all document frequencies, as if a single document containing
              every unique term was added to the corpus.
            - "bm25": idf = log((n_docs - df + 0.5) / (df + 0.5)), which is
              a form commonly used in information retrieval that allows for
              very common terms to receive negative weights.
            - None: no global weighting is applied to local term weights.

        dl_type: Type of document-length scaling to use for weights'
            normalization component:

            - "linear": dl (dls are already linear, so left as-is)
            - "sqrt": dl => sqrt(dl)
            - "log": dl => log(dl)
            - None: no normalization is applied to local(*global?) weights

        norm: If "l1" or "l2", normalize weights by the L1 or L2 norms, respectively,
            of row-wise vectors; otherwise, don't.
        min_df: Minimum number of documents in which a term must appear for it to be
            included in the vocabulary and as a column in a transformed doc-term matrix.
            If float, value is the fractional proportion of the total number of docs,
            which must be in [0.0, 1.0]; if int, value is the absolute number.
        max_df: Maximum number of documents in which a term may appear for it to be
            included in the vocabulary and as a column in a transformed doc-term matrix.
            If float, value is the fractional proportion of the total number of docs,
            which must be in [0.0, 1.0]; if int, value is the absolute number.
        max_n_terms: If specified, only include terms whose document frequency is within
            the top ``max_n_terms``.
        vocabulary_terms: Mapping of unique term string to unique term id, or
            an iterable of term strings that gets converted into such a mapping.
            Note that, if specified, vectorized outputs will include *only* these terms.

        “lucene-style tfidf”: Adds a doc-length normalization to the usual local and global components.
            Params: tf_type="linear", apply_idf=True, idf_type="smooth", apply_dl=True, dl_type="sqrt"

        “lucene-style bm25”: Uses a smoothed idf instead of the classic bm25 variant to prevent weights on terms from going negative.
            Params: tf_type="bm25", apply_idf=True, idf_type="smooth", apply_dl=True, dl_type="linear"
    Attributes:
        doc_term_matrix
    Returns:
        transform_fit'ted vectorizer
    """
    # make sure tokenized_docs is the right typing
    try:
        for xelm in iter(tokenized_docs):
            for elm in iter(xelm):
                assert isinstance(elm, str)
    except AssertionError:
        raise AssertionError(" tokenized_docs is not of the typing  Iterable[Iterable[str]] ")
    except Exception as e:
        logger.error(e)
        raise

    vectorizer = Vectorizer(
        # tf_type="linear", idf_type="smooth", norm="l2",  min_df=3, max_df=0.95)
        tf_type=tf_type,
        idf_type=idf_type,
        dl_type=dl_type,
        norm=norm,
        min_df=min_df,
        max_df=max_df,
        max_n_terms=max_n_terms,
        vocabulary_terms=vocabulary_terms
    )
    doc_term_matrix = vectorizer.fit_transform(tokenized_docs)

    gen_model.doc_term_matrix = doc_term_matrix

    return vectorizer
