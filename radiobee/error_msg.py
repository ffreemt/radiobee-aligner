"""Prepare an error message for gradiobee."""
from typing import Optional, Tuple, Union
import pandas as pd


def error_msg(
    msg: Optional[Union[str, Exception]],
    title: str = "error message",
) -> Tuple[Union[pd.DataFrame, None], None, None, None, None, None]:
    """Prepare an error message for gradiobee outputs."""
    if msg is None:
        msg = "none..."

    try:
        msg = msg.__str__()
    except Exception as exc:
        msg = str(exc)

    df = pd.DataFrame([msg], columns=[title])

    # return df, *((None,) * 4)  # pyright complains
    return df, None, None, None, None, None
