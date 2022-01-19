"""Interpolate np.nan."""
# pylint: disable=invalid-name
from typing import List, Tuple
import numpy as np
import pandas as pd


# fmt: off
def interpolate_pset(
        pairs: List[Tuple[int, int, float]],
        tgt_len: int,
        method: str = 'linear',
        limit_direction: str = 'both',
) -> List[Tuple[int, int]]:
    # fmt: on
    """Interpolate.

    Args:
        pairs: integer pairs, some np.nan
        tgt_len: over 0...tgt_len-1 (x-axis, cmat.shape[1])
        method: for use in pd.DataFrame.interpolate
        limit_direction:  for use in pd.DataFrame.interpolate
    Returns:
        np.nan converted
    """
    y00, *_ = zip(*pairs)

    res = []
    for idx in range(tgt_len):
        if idx in y00:
            loc = y00.index(idx)
            res.append(tuple(pairs[loc][:2]))
        else:
            res.append((idx, np.nan))

    df = pd.DataFrame(res, columns=["y00", "yargmax"])
    _ = df.interpolate(method=method, limit_direction=limit_direction, axis=0)

    _ = _.to_numpy(dtype=int)
    _ = [(int(elm0), int(elm1)) for elm0, elm1 in _]

    return _
