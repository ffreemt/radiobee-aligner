"""Plot pandas.DataFrame with DBSCAN clustering."""
# pylint: disable=invalid-name, too-many-arguments
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN

from fastlid import fastlid
import logzero
from logzero import logger

from radiobee.cmat2tset import cmat2tset

# turn interactive when in ipython session
_ = """
if "get_ipython" in globals():
    plt.ion()
else:
    plt.switch_backend("Agg")
# """

logzero.loglevel(20)  # 10: debug on
fastlid.set_languages = ["en", "zh"]


# fmt: off
def plot_cmat(
        # df_: pd.DataFrame,
        cmat: np.ndarray,
        eps: float = 10,
        min_samples: int = 6,
        # ylim: int = None,
        xlabel: str = "zh",
        ylabel: str = "en",
        backend: str = "Agg",
        showfig: bool = False,
):
    # ) -> plt:
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
    logger.debug(
        '"get_ipython" in globals(): %s', "get_ipython" in globals()
    )

    len1, len2 = cmat.shape

    df_ = pd.DataFrame(cmat2tset(cmat))
    df_.columns = ["x", "y", "cos"]

    backend_saved = matplotlib.get_backend()

    # switch backend if necessary
    if backend_saved != backend:
        plt.switch_backend(backend)

    # len1 = len(lst1)  # noqa
    # len2 = len(lst2)  # noqa

    # lang1, _ = fastlid(" ".join(lst1))
    # lang2, _ = fastlid(" ".join(lst2))
    # xlabel: str = lang1
    # ylabel: str = lang2

    sns.set()
    sns.set_style("darkgrid")

    # close all existing figures, necesssary for hf spaces
    plt.close("all")
    # if sys.platform not in ["win32", "linux"]:
    # plt.switch_backend('Agg')  # to cater for Mac, thanks to WhiteFox

    fig = plt.figure()
    gs = fig.add_gridspec(2, 2, wspace=0.4, hspace=0.58)
    ax2 = fig.add_subplot(gs[0, 0])
    ax0 = fig.add_subplot(gs[0, 1])
    ax1 = fig.add_subplot(gs[1, 0])

    cmap = "viridis_r"
    sns.heatmap(cmat, cmap=cmap, ax=ax2).invert_yaxis()
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)
    ax2.set_title("cos similarity heatmap")

    fig.suptitle("alignment projection")

    _ = DBSCAN(min_samples=min_samples, eps=eps).fit(df_).labels_ > -1
    # _x = DBSCAN(min_samples=min_samples, eps=eps).fit(df_).labels_ < 0
    _x = ~_

    df_.plot.scatter("x", "y", c="cos", cmap=cmap, ax=ax0)

    # clustered
    df_[_].plot.scatter("x", "y", c="cos", cmap=cmap, ax=ax1)

    # outliers
    df_[_x].plot.scatter("x", "y", c="r", marker="x", alpha=0.6, ax=ax0)

    ax0.set_xlabel(xlabel)
    ax0.set_ylabel(ylabel)

    ax0.set_xlim(0, len1)
    ax0.set_ylim(0, len2)
    ax0.set_title("max along columns ('x': outliers)")

    # ax1.set_xlabel("en")
    # ax1.set_ylabel("zh")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)

    ax1.set_xlim(0, len1)
    ax1.set_ylim(0, len2)
    ax1.set_title(f"potential aligned pairs ({round(sum(_) / len1, 2):.0%})")

    logger.debug(" matplotlib.get_backend(): %s", matplotlib.get_backend())

    # if matplotlib.get_backend() not in ["Agg"]:
    if showfig:
        # plt.ioff()  # or we'll just see the plot show and disappear
        # plt.show()
        plt.show(block=True)

    # restore if necessary
    if backend_saved != backend:
        plt.switch_backend(backend_saved)

    # return plt
