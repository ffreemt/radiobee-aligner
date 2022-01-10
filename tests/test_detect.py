"""Test detect."""
import pytest
from radiobee.detect import detect


@pytest.mark.parametrize(
    "test_input,expected", [
        ("", "en"),
        (" ", "en"),
        (" \n ", "en"),
        ("注释", "zh"),
    ]
)
def test_detect(test_input, expected):
    """Test detect."""
    assert detect(test_input) == expected

    # expected set_languages[0], set_languages = ["en", "zh"]
    assert detect(test_input, ["en", "zh"]) == expected


def test_detect_de():
    """Test detect de."""
    text = "4\u3000In der Beschränkung zeigt sich erst der Meister, / Und das Gesetz nur kann uns Freiheit geben. 参见http://www.business-it.nl/files/7d413a5dca62fc735a072b16fbf050b1-27.php."  # noqa
    assert detect(text) == "de"
    assert detect(text, ["en", "zh"]) == "zh"
