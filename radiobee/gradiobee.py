"""Gradiobee."""
from pathlib import Path
from itertools import zip_longest

# import tempfile

from logzero import logger
from sklearn.cluster import DBSCAN
from fastlid import fastlid

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# from radiobee.process_upload import process_upload
from radiobee.files2df import files2df
from radiobee.file2text import file2text
from radiobee.lists2cmat import lists2cmat
from radiobee.gen_pset import gen_pset
from radiobee.gen_aset import gen_aset
from radiobee.align_texts import align_texts
from radiobee.cmat2tset import cmat2tset
from radiobee.trim_df import trim_df
from radiobee.error_msg import error_msg
from radiobee.text2lists import text2lists

sns.set()
sns.set_style("darkgrid")

debug = False
debug = True


def gradiobee(
    file1,
    file2,
    tf_type,
    idf_type,
    dl_type,
    norm,
    eps,
    min_samples,
    # debug=False,
):
    """Process inputs and return outputs."""
    logger.debug(" *debug* ")

    # possible further switchse
    # para_sent: para/sent
    # sent_ali: default/radio/gale-church
    plot_dia = True  # noqa

    # outputs: check return
    # if outputs is modified, also need to modify error_msg's outputs

    # convert "None" to None for those Radio types
    for _ in [idf_type, dl_type, norm]:
        if _ in "None":
            _ = None

    # logger.info("file1: *%s*, file2: *%s*", file1, file2)
    if file2 is not None:
        logger.info("file1.name: *%s*, file2.name: *%s*", file1.name, file2.name)
    else:
        logger.info("file1.name: *%s*, file2: *%s*", file1.name, file2)

    # bypass if file1 or file2 is str input
    # if not (isinstance(file1, str) or isinstance(file2, str)):
    text1 = file2text(file1)

    if file2 is None:
        logger.debug("file2 is None")
        text2 = ""

        # TODO split text1 to text1 and text2
    else:
        logger.debug("file2.name: %s", file2.name)
        text2 = file2text(file2)

    # if not text1.strip() or not text2.strip():
    if not text1.strip():
        msg = (
            "file 1 is apparently empty... Upload a none empty file and try again."
            # f"text1[:10]: [{text1[:10]}], "
            # f"text2[:10]: [{text2[:10]}]"
        )
        return error_msg(msg)

    # single file
    # when text2 is empty
    # process file1/text1: split text1 to text1 text2 to zh-en
    if not text2.strip():
        _ = [elm.strip() for elm in text1.splitlines() if elm.strip()]
        if not _:  # essentially empty file1
            return error_msg("Nothing worthy of processing in file 1")

        _ = zip_longest(_, [""])
        _ = pd.DataFrame(_, columns=["text1", "text2"])
        df_trimmed = trim_df(_)

        # text1 = loadtext("data/test-dual.txt")
        list1, list2 = text2lists(text1)

        lang1 = text2lists.lang1
        lang2 = text2lists.lang2
        offset = text2lists.offset  # noqa

        _ = """
        ax = sns.heatmap(lists2cmat(list1, list2), cmap="gist_earth_r")  # ax=plt.gca()
        ax.invert_yaxis()
        ax.set(
            xlabel=lang1,
            ylabel=lang2,
            title=f"cos similary heatmap \n(offset={offset})",
        )
        plt_loc = "img/plt.png"
        plt.savefig(plt_loc)
        # """

        # output_plot = plt_loc  # for gr.outputs.Image

        #
        _ = zip_longest(list1, list2, fillvalue="")
        df_aligned = pd.DataFrame(
            _,
            columns=["text1", "tex2"]
        )

        file_dl = Path(f"{Path(file1.name).stem[:-8]}-{lang1}-{lang2}.csv")
        file_dl_xlsx = Path(
            f"{Path(file1.name).stem[:-8]}-{lang1}-{lang2}.xlsx"
        )

        # return  df_trimmed, output_plot, file_dl, file_dl_xlsx, df_aligned

    # end if single file
    else:  # file1 file 2: proceed
        lang1, _ = fastlid(text1)
        lang2, _ = fastlid(text2)

        df1 = files2df(file1, file2)

        list1 = [elm for elm in df1.text1 if elm]
        list2 = [elm for elm in df1.text2 if elm]
        # len1 = len(list1)  # noqa
        # len2 = len(list2)  # noqa

        file_dl = Path(f"{Path(file1.name).stem[:-8]}-{Path(file2.name).stem[:-8]}.csv")
        file_dl_xlsx = Path(
            f"{Path(file1.name).stem[:-8]}-{Path(file2.name).stem[:-8]}.xlsx"
        )

        df_trimmed = trim_df(df1)
    # --- end else single

    try:
        cmat = lists2cmat(
            list1,
            list2,
            tf_type=tf_type,
            idf_type=idf_type,
            dl_type=dl_type,
            norm=norm,
        )
    except Exception as exc:
        logger.error(exc)
        return error_msg(exc)

    tset = pd.DataFrame(cmat2tset(cmat))
    tset.columns = ["x", "y", "cos"]

    _ = """
    df_trimmed = pd.concat(
        [
            df1.iloc[:4, :],
            pd.DataFrame(
                [
                    [
                        "...",
                        "...",
                    ]
                ],
                columns=df1.columns,
            ),
            df1.iloc[-4:, :],
        ],
        ignore_index=1,
    )
    # """

    # process list1, list2 to obtained df_aligned
    # quick fix ValueError: not enough values to unpack (expected at least 1, got 0)
    # fixed in gen_pet, but we leave the loop here
    for min_s in range(min_samples):
        logger.info(" min_samples, using %s", min_samples - min_s)
        try:
            pset = gen_pset(
                cmat,
                eps=eps,
                min_samples=min_samples - min_s,
                delta=7,
            )
            break
        except ValueError:
            logger.info(" decrease min_samples by %s", min_s + 1)
            continue
        except Exception as e:
            logger.error(e)
            continue
    else:
        # break should happen above when min_samples = 2
        raise Exception("bummer, this shouldn't happen, probably another bug")

    min_samples = gen_pset.min_samples

    # will result in error message:
    # UserWarning: Starting a Matplotlib GUI outside of
    # the main thread will likely fail."
    _ = """
    plot_cmat(
        cmat,
        eps=eps,
        min_samples=min_samples,
        xlabel=lang1,
        ylabel=lang2,
    )
    # """

    # move plot_cmat's code to the main thread here
    # to make it work
    xlabel = lang1
    ylabel = lang2

    len1, len2 = cmat.shape
    ylim, xlim = len1, len2

    # does not seem to show up
    logger.debug(" len1 (ylim): %s, len2 (xlim): %s", len1, len2)
    if debug:
        print(f" len1 (ylim): {len1}, len2 (xlim): {len2}")

    df_ = pd.DataFrame(cmat2tset(cmat))
    df_.columns = ["x", "y", "cos"]

    sns.set()
    sns.set_style("darkgrid")

    # close all existing figures, necesssary for hf spaces
    plt.close("all")

    # if sys.platform not in ["win32", "linux"]:
    # going for noninterative
    # to cater for Mac, thanks to WhiteFox
    plt.switch_backend("Agg")

    # figsize=(13, 8), (339, 212) mm on '1280x800+0+0'
    fig = plt.figure(figsize=(13, 8))

    # gs = fig.add_gridspec(2, 2, wspace=0.4, hspace=0.58)
    gs = fig.add_gridspec(1, 2, wspace=0.4, hspace=0.58)
    ax_heatmap = fig.add_subplot(gs[0, 0])  # ax2
    ax0 = fig.add_subplot(gs[0, 1])
    # ax1 = fig.add_subplot(gs[1, 0])

    cmap = "viridis_r"
    sns.heatmap(cmat, cmap=cmap, ax=ax_heatmap).invert_yaxis()
    ax_heatmap.set_xlabel(xlabel)
    ax_heatmap.set_ylabel(ylabel)
    ax_heatmap.set_title("cos similarity heatmap")

    fig.suptitle(f"alignment projection\n(eps={eps}, min_samples={min_samples})")

    _ = DBSCAN(min_samples=min_samples, eps=eps).fit(df_).labels_ > -1
    # _x = DBSCAN(min_samples=min_samples, eps=eps).fit(df_).labels_ < 0
    _x = ~_

    # max cos along columns
    df_.plot.scatter("x", "y", c="cos", cmap=cmap, ax=ax0)

    # outliers
    df_[_x].plot.scatter("x", "y", c="r", marker="x", alpha=0.6, ax=ax0)
    ax0.set_xlabel(xlabel)
    ax0.set_ylabel(ylabel)
    ax0.set_xlim(xmin=0, xmax=xlim)
    ax0.set_ylim(ymin=0, ymax=ylim)
    ax0.set_title(
        "max along columns (x: outliers)\n"
        "potential aligned pairs (green line), "
        f"{round(sum(_) / xlim, 2):.0%}"
    )

    plt_loc = "img/plt.png"
    plt.savefig(plt_loc)

    # clustered
    # df_[_].plot.scatter("x", "y", c="cos", cmap=cmap, ax=ax1)
    # ax1.set_xlabel(xlabel)
    # ax1.set_ylabel(ylabel)
    # ax1.set_xlim(0, len1)
    # ax1.set_title(f"potential aligned pairs ({round(sum(_) / len1, 2):.0%})")
    # end of plot_cmat

    src_len, tgt_len = cmat.shape
    aset = gen_aset(pset, src_len, tgt_len)
    final_list = align_texts(aset, list2, list1)  # note the order

    # df_aligned
    df_aligned = pd.DataFrame(final_list, columns=["text1", "text2", "likelihood"])

    # swap text1 text2
    df_aligned = df_aligned[["text2", "text1", "likelihood"]]
    df_aligned.columns = ["text1", "text2", "likelihood"]

    # ===
    if plot_dia:
        output_plot = "img/plt.png"
    else:
        output_plot = None

    _ = df_aligned.to_csv(index=False)
    file_dl.write_text(_, encoding="utf8")

    # file_dl.write_text(_, encoding="gb2312")  # no go
    df_aligned.to_excel(file_dl_xlsx)

    # return df_trimmed, plt

    # return df_trimmed, plt, file_dl, file_dl_xlsx, df_aligned

    # output_plot: gr.outputs.Image(type="auto", label="...")
    return df_trimmed, output_plot, file_dl, file_dl_xlsx, df_aligned

    # modi outputs