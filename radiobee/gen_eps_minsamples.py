"""Gen suggested eps min_samples."""


def gen_eps_minsamples(src_len: int, tgt_len: int) -> dict:
    """Gen suggested eps min_samples."""
    eps = src_len * 0.01

    # if eps < 3: eps = 3
    eps = max(3, eps)

    min_samples = tgt_len / 100 * 0.5

    # if min_samples < 3: min_samples = 3
    min_samples = max(3, min_samples)

    return {"eps": eps, "min_samples": min_samples}
