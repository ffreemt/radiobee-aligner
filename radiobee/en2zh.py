"""Translate english to chinese via a dict."""
# from typing import List, Union
from typing import Iterable, List, Union

import warnings

import copy
from radiobee.mdx_e2c import mdx_e2c

warnings.simplefilter('ignore', DeprecationWarning)


# fmt: off
def en2zh(
        # text: Union[str, List[List[str]]],
        # text: Union[str, List[str]],
        text: Union[str, Iterable[str]],
) -> List[str]:
    # fmt: on
    """Translate english to chinese via a dict.

    Args
        text: to translate, list of str

    Returns
        res: list of str
    """
    res = copy.deepcopy(text)
    if isinstance(text, str):
        # res = [text.split()]
        res = [text]

    # if res and isinstance(res[0], str):
        # res = [line.lower().split() for line in res]

    # res = ["".join([word_tr(word) for word in line]) for line in res]
    _ = []
    for line in res:
        line_tr = [mdx_e2c(word) for word in line.split()]
        _.append("".join(line_tr))

    return _
