"""Test loadtext."""
import pytest

from radiobee.loadtext import loadtext


def test_loadtext():
    _ = loadtext("data/test_en.txt").splitlines()
    _ = [elm for elm in _ if elm.strip()]
    assert len(_) == 33


@pytest.mark.xfail
def test_loadtext_from_dir():
    """Test test_loadtext_from_dir."""
    _ = loadtext(".")
