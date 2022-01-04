"""gen tokens for english or chinese text for a given model."""
# pylint: disable=

from typing import List

from textacy.representations import Vectorizer
from radiobee.insert_spaces import insert_spaces
# from radiobee.gen_model import gen_model


def gen_vector(text: str, model: Vectorizer) -> List[float]:
    """Gen vector for a give model.

    Args:
        text: string of Chinese chars or English words.
    """
    vec = insert_spaces(text).split()

    return model.transform(vec)
