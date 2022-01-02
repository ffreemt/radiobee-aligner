"""Gne pset from cmat. Find pairs for a given cmat.

tinybee.find_pairs.py with fixed estimator='dbscan' eps=eps, min_samples=min_samples
"""
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import logzero
from logzero import logger
from radiobee.cmat2tset import cmat2tset
from radiobee.interpolate_pset import interpolate_pset


def _gen_pset(
    cmat1: Union[List[List[float]], np.ndarray, pd.DataFrame],
    eps: float = 10,
    min_samples: int = 6,
    delta: float = 7,
    verbose: Union[bool, int] = False,
    # ) -> List[Tuple[int, int, Union[float, str]]]:
) -> List[Tuple[Union[float, str], Union[float, str], Union[float, str]]]:
    """Gen pset from cmat.
    Find pairs for a given cmat.

    Args:
        cmat: correlation/similarity matrix
        eps: min epsilon for DBSCAN (10)
        min_samples: minimum # of samples for DBSCAN (6)
        delta: tolerance (7)

    Returns:
        pairs + "" or metric (float)

    dbscan_pairs' setup
        if eps is None:
            eps = src_len * .01
            if eps < 3:
                eps = 3
        if min_samples is None:
            min_samples = tgt_len / 100 * 0.5
            if min_samples < 3:
                min_samples = 3

    def gen_eps_minsamples(src_len, tgt_len):
        eps = src_len * .01
        if eps < 3:
            eps = 3

        min_samples = tgt_len / 100 * 0.5
        if min_samples < 3:
            min_samples = 3
        return {"eps": eps, "min_samples": min_samples}

    """
    if isinstance(verbose, bool):
        if verbose:
            verbose = 10
        else:
            verbose = 20
    logzero.loglevel(verbose)

    # if isinstance(cmat, list):
    cmat = np.array(cmat1)

    src_len, tgt_len = cmat.shape

    # tset = cmat2tset(cmat)
    tset = cmat2tset(cmat).tolist()

    logger.debug("tset: %s", tset)

    # iset = gen_iset(cmat, verbose=verbose, estimator=estimator)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(tset).labels_

    df_tset = pd.DataFrame(tset, columns=["x", "y", "cos"])
    cset = df_tset[labels > -1].to_numpy()

    # sort cset
    _ = sorted(cset.tolist(), key=lambda x: x[0])
    iset = interpolate_pset(_, tgt_len)

    # *_, ymax = zip(*tset)
    # ymax = list(ymax)
    # low_ = np.min(ymax) - 1  # reset to minimum_value - 1

    buff = [(-1, -1, ""), (tgt_len, src_len, "")]
    # for _ in range(tgt_len):
    for idx, tset_elm in enumerate(tset):
        logger.debug("buff: %s", buff)
        # postion max in ymax and insert in buff
        # if with range given by iset+-delta and
        # it's valid (do not exceed constraint
        # by neighboring points

        # argmax = int(np.argmax(ymax))

        # logger.debug("=== %s,%s === %s", _, argmax, tset[_])
        logger.debug("=== %s === %s", _, tset_elm)

        # ymax[_] = low_
        # elm = tset[argmax]
        # elm0, *_ = elm

        elm0, *_ = tset_elm

        # position elm in buff
        idx = -1  # for making pyright happy
        for idx, loc in enumerate(buff):
            if loc[0] > elm0:
                break
        else:
            idx += 1  # last

        # insert elm in for valid elm
        # (within range inside two neighboring points)

        # pos = int(tset[argmax][0])
        pos = int(tset_elm[0])
        logger.debug(" %s <=> %s ", tset_elm, iset[pos])

        # if abs(tset[argmax][1] - iset[pos][1]) <= delta:
        if abs(tset_elm[1] - iset[pos][1]) <= delta:
            if tset_elm[1] > buff[idx - 1][1] and tset_elm[1] < buff[idx][1]:
                buff.insert(idx, tset_elm)
                logger.debug("idx: %s, tset_elm: %s", idx, tset_elm)
            else:
                logger.debug("\t***\t idx: %s, tset_elm: %s", idx, tset_elm)
        _ = """
        if abs(tset[loc][1] - iset[loc][1]) <= delta:
            if tset[loc][1] > buff[idx][1] and tset[loc][1] < buff[idx + 1][1]:
                buff.insert(idx + 1, tset[loc])
        # """

    # remove first and last entry in buff
    buff.pop(0)
    buff.pop()

    # return [(1, 1, "")]
    return [(int(elm0), int(elm1), elm2) for elm0, elm1, elm2 in buff]


def gen_pset(
    cmat1: Union[List[List[float]], np.ndarray, pd.DataFrame],
    eps: float = 10,
    min_samples: int = 6,
    delta: float = 7,
    verbose: Union[bool, int] = False,
) -> List[Tuple[Union[float, str], Union[float, str], Union[float, str]]]:
    """Gen pset.

    Refer to _gen_pset.
    """
    gen_pset.min_samples = min_samples
    for min_s in range(min_samples):
        logger.debug(" min_samples, try %s", min_samples - min_s)
        try:
            pset = _gen_pset(
                cmat1,
                eps=eps,
                min_samples=min_samples - min_s,
                delta=delta,
            )
            break
        except ValueError:
            logger.debug(" decrease min_samples by %s", min_s + 1)
            continue
        except Exception as e:
            logger.error(e)
            continue
    else:
        # break should happen above when min_samples = 2
        raise Exception("bummer, this shouldn't happen, probably another bug")

    # store new min_samples
    gen_pset.min_samples = min_samples - min_s

    return pset
