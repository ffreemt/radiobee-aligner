"""Gen proper alignment for a given triple_set.

cmat = fetch_sent_corr(src, tgt)
src_len, tgt_len = np.array(cmat).shape
r_ali = gen_row_alignment(cmat, tgt_len, src_len)  # note the order
src[r_ali[1]], tgt[r_ali[0]], r_ali[2]

or  !!!  (targer, source)
cmat = fetch_sent_corr(tgt, src)  # note the order
src_len, tgt_len = np.array(cmat).shape
r_ali = gen_row_alignment(cmat, src_len, tgt_len)
src[r_ali[0]], tgt[r_ali[1]], r_ali[2]

---
src_txt = 'data/wu_ch2_en.txt'
tgt_txt = 'data/wu_ch2_zh.txt'

assert Path(src_txt).exists()
assert Path(tgt_txt).exists()

src_text, _ = load_paras(src_txt)
tgt_text, _ = load_paras(tgt_txt)

cos_matrix = gen_cos_matrix(src_text, tgt_text)
t_set, m_matrix = find_aligned_pairs(cos_matrix0, thr=0.4, matrix=True)

resu = gen_row_alignment(t_set, src_len, tgt_len)
resu = np.array(resu)

idx = -1
idx += 1; (resu[idx], src_text[int(resu[idx, 0])],
    tgt_text[int(resu[idx, 1])]) if all(resu[idx]) else resu[idx]

idx += 1;  i0, i1, i2 = resu[idx]; '***' if i0 == ''
else src_text[int(i0)], '***' if i1 == '' else tgt_text[int(i1)], ''
if i2 == '' else i2
"""
# pylint: disable=line-too-long, unused-variable
from typing import List, Union

# natural extrapolation with slope equal to 1
from itertools import zip_longest as zip_longest_middle

import numpy as np

from logzero import logger

# from tinybee.zip_longest_middle import zip_longest_middle

# from tinybee.zip_longest_middle import zip_longest_middle
# from tinybee.find_pairs import find_pairs

# logger = logging.getLogger(__name__)
# logger.addHandler(logging.NullHandler())


def gen_row_alignment(  # pylint: disable=too-many-locals
    t_set,
    src_len,
    tgt_len,
    # ) -> List[Tuple[Union[str, int], Union[str, int], Union[str, float]]]:
) -> List[List[Union[str, float]]]:
    """Gen proper rows for given triple_set.

    Arguments:
        [t_set {np.array or list}] -- [nll matrix]
        [src_len {int}] -- numb of source texts (para/sents)
        [tgt_len {int}] -- numb of target texts (para/sents)

    Returns:
        [np.array] -- [proper rows]
    """
    t_set = np.array(t_set, dtype="object")

    # len0 = src_len

    # len1 tgt text length, must be provided
    len1 = tgt_len

    # rearrange t_set as buff in increasing order
    buff = [[-1, -1, ""]]  #
    idx_t = 0
    # for elm in t_set:
    # start with bigger value from the 3rd col

    y00, yargmax, ymax = zip(*t_set)
    ymax_ = np.array(ymax).copy()
    reset_v = np.min(ymax_) - 1
    for count in range(tgt_len):
        argmax = np.argmax(ymax_)
        # reset
        ymax_[argmax] = reset_v
        idx_t = argmax
        elm = t_set[idx_t]
        logger.debug("%s: %s, %s", count, idx_t, elm)

        # find loc to insert
        elm0, elm1, elm2 = elm
        idx = -1
        for idx, loc in enumerate(buff):
            if loc[0] > elm0:
                break
        else:
            idx += 1  # last

        # make sure elm1 is within the range
        # prev elm1 < elm1 < next elm1
        if elm1 > buff[idx - 1][1]:
            try:  # overflow possible (idx + 1 in # last)
                next_elm = buff[idx][1]
            except IndexError:
                next_elm = len1
            if elm1 < next_elm:
                # insert '' if necessary
                # using zip_longest_middle
                buff.insert(
                    idx, [elm0, elm1, elm2],
                )
                # logger.debug('---')

        idx_t += 1
        # if idx_t == 24:  # 20:
        #     break

    # remove [-1, -1]
    # buff.pop(0)
    # buff = np.array(buff, dtype='object')

    # take care of the tail
    buff += [[src_len, tgt_len, ""]]

    resu = []
    # merit = []

    for idx, elm in enumerate(buff[1:]):
        idx1 = idx + 1
        elm0_, elm1_, elm2_ = buff[idx1 - 1]  # idx starts from 0
        elm0, elm1, elm2 = elm
        del elm2_, elm2

        tmp0 = zip_longest_middle(
            list(range(elm0_ + 1, elm0)), list(range(elm1_ + 1, elm1)), fillvalue="",
        )
        # convet to list entries & attache merit
        tmp = [list(t_elm) + [""] for t_elm in tmp0]

        # update resu
        resu += tmp + [buff[idx1]]

    # remove the last entry
    return resu[:-1]
