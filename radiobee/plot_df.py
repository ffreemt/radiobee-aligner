"""Plot pandas.DataFrame with DBSCAN clustering."""
# pylint: disable=invalid-name, too-many-arguments
import numpy as np  # noqa
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN

from logzero import logger  # noqa

# from radiobee.cmat2tset import cmat2tset

# turn interactive when in ipython session
_ = """
if "get_ipython" in globals():
    plt.ion()
else:
    plt.switch_backend('Agg')
# """
# fastlid.set_languages = ["en", "zh"]


# fmt: off
def plot_df(
        df_: pd.DataFrame,
        # cmat: np.ndarray,
        eps: float = 10,
        min_samples: int = 6,
        xlabel: str = "",
        ylabel: str = "",
        xlim: int = 0,
        ylim: int = 0,
        backend: str = "TkAgg",
) -> plt:
    # fmt: on
    """Plot df with DBSCAN clustering.

    Args:
        df_: pandas.DataFrame, with three columns columns=["x", "y", "cos"]
    Returns:
        matplotlib.pyplot: for possible use in gradio

    plot_df(pd.DataFrame(cmat2tset(smat), columns=['x', 'y', 'cos']))
    df_ = pd.DataFrame(cmat2tset(smat), columns=['x', 'y', 'cos'])

    # sort 'x', axis 0 changes, index regenerated
    df_s = df_.sort_values('x', axis=0, ignore_index=True)

    # sorintg does not seem to impact clustering
    DBSCAN(1.5, min_samples=3).fit(df_).labels_
    DBSCAN(1.5, min_samples=3).fit(df_s).labels_

    """
    # df_ = pd.DataFrame(cmat2tset(cmat))
    if df_.shape[1] == 3:
        df_.columns = ["x", "y", "cos"]
    else:
        logger.error(" shape mismatch: %s, expected (x, 3)", df_.shape)
        # return None
        raise Exception(" df_.shape[1] not equal to 3 ")

    if not xlim:
        xlim = len(df_)
    if not ylim:
        ylim = df_.y.max()

    if not xlabel:
        xlabel = str(xlim)
    if not ylabel:
        ylabel = str(ylim)

    backend_saved = matplotlib.get_backend()

    # switch if necessary
    if backend_saved != backend:
        plt.switch_backend(backend)

    sns.set()
    sns.set_style("darkgrid")

    fig = plt.figure(figsize=(13, 8))

    # gs = fig.add_gridspec(2, 2, wspace=0.4, hspace=0.58)
    # ax2 = fig.add_subplot(gs[0, 0])
    # ax0 = fig.add_subplot(gs[0, 1])
    # ax1 = fig.add_subplot(gs[1, 0])

    gs = fig.add_gridspec(1, 1, wspace=0.4, hspace=0.58)
    ax0 = fig.add_subplot(gs[0, 0])

    cmap = "viridis_r"

    _ = DBSCAN(min_samples=min_samples, eps=eps).fit(df_).labels_ > -1
    _x = ~_

    # clustered
    df_[_].plot.scatter("x", "y", c="cos", cmap=cmap, ax=ax0)
    # outliers
    df_[_x].plot.scatter("x", "y", c="r", marker="x", alpha=0.6, ax=ax0)

    # ax1.set_xlabel("en")
    # ax1.set_ylabel("zh")
    ax0.set_xlabel(xlabel)
    ax0.set_ylabel(ylabel)

    # ax0.set_xlim(0, xlim)
    # ax0.set_ylim(0, ylim)
    ax0.set_title("max cos ('x': outliers)")

    # ax1.set_title(f"potential aligned pairs ({round(sum(_) / xlim, 2):.0%})")

    # restore if necessary
    if backend_saved != backend:
        plt.switch_backend(backend_saved)

    return plt


_ = """
        eps: float = 10
        min_samples: int = 6
        xlabel: str = ""
        ylabel: str = ""
        xlim: int = 0
        ylim: int = 0
"""
