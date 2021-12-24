"""Load mdx_dict_e2c c2e.

mdx_e2c = joblib.load("./mdx_dict_e2c.lzma")
mdx_c2e = joblib.load("./mdx_dict_e2c.lzma")
"""
from pathlib import Path
from string import punctuation
import joblib

# keep "-"
punctuation = punctuation.replace("-", "")
c_dir = Path(__file__).parent

# lazy load in __init__.py like this?
# mdx_dict_e2c = importlib.import_module("mdx_dict_e2c")
# mdx_e2c = mdx_dict_e2c.mdx_e2c
# mdx_dict_c2e = importlib.import_module("mdx_dict_c2e")
# mdx_c2e = mdx_dict_c2e.mdx_c2e

mdx_dict_e2c = joblib.load(c_dir / "mdx_dict_e2c.lzma")
print("e2c lzma file loaded")

# memory = joblib.Memory("joblibcache", verbose=0)


# @memory.cache  # no need, mdx_dict_e2c in RAM
def mdx_e2c(word: str) -> str:
    """Fetch definition for word.

    Args:
        word: word to look up
    Returns:
        definition entry or word itself
    >>> mdx_e2c("do").__len__()
    43
    >>> mdx_e2c("我").strip()
    '我'
    """
    word = word.strip(punctuation + " \t\n\r")
    return mdx_dict_e2c.get(word.lower(), word)
