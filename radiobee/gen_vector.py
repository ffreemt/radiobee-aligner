"""gen tokens for english or chinese text for a given model."""
# pylint: disable=

from typing import List, Union

from textacy.representations import Vectorizer
from radiobee.insert_spaces import insert_spaces
# from radiobee.gen_model import gen_model


def gen_vector(text: Union[str, List[str]], model: Vectorizer) -> List[float]:
    """Gen vector for a give model.

    Args:
        text: string of Chinese chars or English words.

    filename = r"data\test-dual.txt"
    text = loadtext(filename)
    list1, list2 = zip(*text2lists(text))
    model = gen_model(list1)
    """
    if isinstance(text, str):
        vec = insert_spaces(text).split()

        return model.transform(vec)

    # already same tokens as used to gen_model
    return model.transform(text)
