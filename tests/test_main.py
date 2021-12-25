"""Test __main__.py."""
# pylint: disable=invalid-name
import tempfile
from fastlid import fastlid

from logzero import logger

# globals()["file2text"] = getattr(importlib.import_module(f"{radiobee.__name__}.file2text"), "file2text")
# from radiobee.process_upload import process_upload  # same as file2text
from radiobee.files2df import files2df
from radiobee.file2text import file2text
from radiobee.lists2cmat import lists2cmat
from radiobee.cmat2tset import cmat2tset

file1loc = "data/test_zh.txt"
file2loc = "data/test_en.txt"

file1 = tempfile._TemporaryFileWrapper(open(file1loc, "rb"), file1loc)
file2 = tempfile._TemporaryFileWrapper(open(file2loc, "rb"), file2loc)

def test_file2file1():
    """Test cmat file2 file1."""
    # logger.info("file1: *%s*, file2: *%s*", file1, file2)
    logger.info("file1.name: *%s*, file2.name: *%s*", file1.name, file2.name)

    text1 = file2text(file1)
    text2 = file2text(file2)

    lang1, _ = fastlid(text1)
    lang2, _ = fastlid(text2)

    lst1 = [elm.strip() for elm in text1.splitlines() if elm.strip()]
    lst2 = [elm.strip() for elm in text2.splitlines() if elm.strip()]

    del lst1, lst2
