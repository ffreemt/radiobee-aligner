"""Plot pandas.DataFrame with DBSCAN clustering."""
# pylint: disable=invalid-name, too-many-arguments
# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN

from logzero import logger

# turn interactive when in ipython session
if "get_ipython" in globals():
    plt.ion()


# fmt: off
def plot_df(
        df_: pd.DataFrame,
        min_samples: int = 6,
        eps: float = 10,
        ylim: int = None,
        xlabel: str = "en",
        ylabel: str = "zh",
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
    df_ = pd.DataFrame(df_)
    if df_.columns.__len__() < 3:
        logger.error(
            "expected 3 columns DataFram, got: %s, cant proceed, returninng None",
            df_.columns.tolist(),
        )
        return None

    # take first three columns
    columns = df_.columns[:3]
    df_ = df_[columns]

    # rename columns to "x", "y", "cos"
    df_.columns = ["x", "y", "cos"]

    sns.set()
    sns.set_style("darkgrid")
    # fig, (ax0, ax1) = plt.subplots(2, figsize=(11.69, 8.27))
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.69, 8.27))

    fig.suptitle("alignment projection")
    _ = DBSCAN(min_samples=min_samples, eps=eps).fit(df_).labels_ > -1
    _x = DBSCAN(min_samples=min_samples, eps=eps).fit(df_).labels_ < 0

    # ax0.scatter(df_[_].x, df_[_].y, marker='o', c='g', alpha=0.5)
    # ax0.grid()
    # print("ratio: %.2f%%" % (100 * sum(_)/len(df_)))

    df_.plot.scatter("x", "y", c="cos", cmap="viridis_r", ax=ax0)

    # clustered
    df_[_].plot.scatter("x", "y", c="cos", cmap="viridis_r", ax=ax1)

    # outliers
    df_[_x].plot.scatter("x", "y", c="r", marker="x", alpha=0.6, ax=ax0)

    # ax0.set_xlabel("")
    # ax0.set_ylabel("zh")
    ax0.set_xlabel("")
    ax0.set_ylabel(ylabel)
    xlim = len(df_)
    ax0.set_xlim(0, xlim)
    if ylim:
        ax0.set_ylim(0, ylim)
    ax0.set_title("max similarity along columns (outliers denoted by 'x')")

    # ax1.set_xlabel("en")
    # ax1.set_ylabel("zh")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)

    ax1.set_xlim(0, xlim)
    if ylim:
        ax1.set_ylim(0, ylim)
    ax1.set_title(f"potential aligned pairs ({round(sum(_) / len(df_), 2):.0%})")

    return plt
