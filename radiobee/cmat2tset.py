"""Gen triple-set from a  matrix."""
from typing import List, Tuple, Union  # noqa

import numpy as np
import pandas as pd


# fmt: off
def cmat2tset(
    cmat1: Union[List[List[float]], np.ndarray, pd.DataFrame],
    # thirdcol: bool = True
) -> np.ndarray:
    # ) -> List[Union[Tuple[int, int], Tuple[int, int, float]]]:
    # fmt: on
    """Gen triple-set from a matrix.

    Args
        cmat: 2d-array or list, correlation or other metric matrix
        # thirdcol: bool, whether to output a third column (max value)

    Returns
        Obtain the max and argmax for each column, erase the row afterwards to eliminate one single row  that would dominate
        every column.
    """
    # if isinstance(cmat, list):
    cmat = np.array(cmat1)

    if not np.prod(cmat.shape):
        raise SystemError("data not 2d...")

    _ = """
    # y00 = range(cmat.shape[1])  # cmat.shape[0] long time wasting bug

    yargmax = cmat.argmax(axis=0)
    if thirdcol:
        ymax = cmat.max(axis=0)

        res = [*zip(y00, yargmax, ymax)]  # type: ignore
        # to unzip
        # a, b, c = zip(*res)

        return res

    _ = [*zip(y00, yargmax)]  # type: ignore
    return _
    """
    low_ = cmat.min() - 1
    argmax_max = []
    src_len, tgt_len = cmat.shape  # ylim, xlim
    for _ in range(min(src_len, tgt_len)):
        argmax = int(cmat.argmax())
        row, col = divmod(argmax, tgt_len)
        argmax_max.append([col, row, cmat.max()])  # x-axis, y-axis

        # erase row-th row and col-th col of cmat
        cmat[row, :] = low_
        cmat[:, col] = low_

    return np.array(argmax_max)
