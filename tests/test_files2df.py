"""Test files2df."""
from pathlib import Path
import tempfile
from radiobee.files2df import files2df


def test_files2df():
    """Test files2df with tests/test_en.txt tests/test_zh.txt."""
    file1_ = "tests/test_en.txt"
    file2_ = "tests/test_zh.txt"
    with open(file1_, 'rb') as fh1, open(file2_, 'rb') as fh2:
        file1 = tempfile._TemporaryFileWrapper(fh1, file1_)
        file2 = tempfile._TemporaryFileWrapper(fh2, file2_)
        assert Path(file1.name).is_file()
        assert Path(file2.name).is_file()

        df = files2df(file1, file2)

    # with filenames as frist row
    # assert df.iloc[1, 0] == "Wuthering Heights"
    # assert df.iloc[1, 1] == "呼啸山庄"

    assert df.iloc[0, 0] == "Wuthering Heights"
    assert df.iloc[0, 1] == "呼啸山庄"


def test_files2df_file2none():
    """Test files2df with tests/test_en.txt None."""
    file1_ = "tests/test_en.txt"
    file2 = None
    with open(file1_, 'rb') as fh1:
        file1 = tempfile._TemporaryFileWrapper(fh1, file1_)
        assert Path(file1.name).is_file()

        df = files2df(file1, file2)

    # with filename as first row
    # assert df.iloc[1, 0] == "Wuthering Heights"
    # assert df.iloc[1, 1] == ""

    assert df.iloc[0, 0] == "Wuthering Heights"
    assert df.iloc[0, 1] == ""
