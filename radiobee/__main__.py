"""Run interactively."""
# pylint: disable=invalid-name, too-many-arguments, unused-argument, redefined-builtin, wrong-import-position, too-many-locals, too-many-statements
from typing import Any, Tuple, Optional, Union  # noqa

import sys
from pathlib import Path
import signal
from random import randint
from textwrap import dedent
from itertools import zip_longest
from socket import socket, AF_INET, SOCK_STREAM

from sklearn.cluster import DBSCAN  # noqa
import joblib
from varname import nameof
from logzero import logger

# import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt  # noqa

# from tabulate import tabulate
from fastlid import fastlid

if "." not in sys.path:
    sys.path.insert(0, ".")
import gradio as gr
from radiobee.process_upload import process_upload
from radiobee.files2df import files2df
from radiobee.file2text import file2text
from radiobee.lists2cmat import lists2cmat
from radiobee.gen_pset import gen_pset
from radiobee.gen_aset import gen_aset
from radiobee.align_texts import align_texts

from radiobee.cmat2tset import cmat2tset

# from radiobee.plot_df import plot_df
# from radiobee.plot_cmat import plot_cmat
from radiobee.trim_df import trim_df

sns.set()
sns.set_style("darkgrid")
fastlid.set_languages = ["en", "zh"]

signal.signal(signal.SIGINT, signal.SIG_DFL)
print("Press Ctrl+C to quit\n")


def savelzma(obj, fileloc: str = None):
    """Aux funciton."""
    if fileloc is None:
        fileloc = nameof(obj)  # this wont work
    joblib.dump(obj, f"data/{fileloc}.lzma")


def greet(input):
    """Greet yo."""
    return f"'Sup yo! (your input: {input})"


def upfile1(file1, file2=None) -> Tuple[str, str]:
    """Upload file1, file2."""
    del file2
    return file1.name, f"'Sup yo! (your input: {input})"


def process_2upoads(file1, file2):
    """Process stuff."""
    # return f"{process_upload(file1)}\n===***\n{process_upload(file2)}"

    text1 = [_.strip() for _ in process_upload(file1).splitlines() if _.strip()]
    text2 = [_.strip() for _ in process_upload(file2).splitlines() if _.strip()]

    text1, text2 = zip(*zip_longest(text1, text2, fillvalue=""))

    df = pd.DataFrame({"text1": text1, "text2": text2})

    # return tabulate(df)
    # return tabulate(df, tablefmt="grid")
    # return tabulate(df, tablefmt='html')

    return df


def error_msg(
    msg: Optional[str],
    title: str = "error message",
    # ) -> Tuple[Union[pd.DataFrame, None], None, None, None, None]:
):
    """Prepare error message for fn outputs."""
    if msg is None:
        msg = "none..."

    try:
        msg = msg.__str__()
    except Exception as exc:
        msg = str(exc)

    df = pd.DataFrame([msg], columns=[title])

    return (df, *((None,) * 4))


if __name__ == "__main__":
    _ = """
    fn = process_2upoads
    inputs = ["file", "file"]
    examples = [
        ["data/test_zh.txt", "data/test_en.txt"],
        ["data/test_en.txt", "data/test_zh.txt"],
    ]
    outputs = ["dataframe"]
    # """
    import logzero

    debug = True
    # debug = False
    if debug:
        logzero.loglevel(10)
    logger.debug(" debug ")
    logger.info(" info ")

    # _ = """
    inputs = [
        gr.inputs.Textbox(
            # placeholder="Input something here",
            default="test text"
        )
    ]
    inputs = ["file", "file"]
    inputs = [
        gr.inputs.File(label="file 1"),
        # gr.inputs.File(file_count="multiple", label="file 2", optional=True),
        gr.inputs.File(label="file 2", optional=True),
    ]

    # modi 1
    _ = """
        tf_type: Literal[linear, sqrt, log, binary] = 'linear'
        idf_type: Optional[Literal[standard, smooth, bm25]] = None
        dl_type: Optional[Literal[linear, sqrt, log]] = None
        norm: norm: Optional[Literal[l1, l2]] = None
        x min_df: int | float = 1
        x max_df: int | float = 1.0
    # """
    input_tf_type = gr.inputs.Dropdown(
        ["linear", "sqrt", "log", "binary"], default="linear"
    )
    input_idf_type = gr.inputs.Radio(
        ["None", "standard", "smooth", "bm25"], default="None"
    )  # need to convert "None" this to None in fn
    input_dl_type = gr.inputs.Radio(
        ["None", "linear", "sqrt", "log"], default="None"
    )  # ditto
    input_norm_type = gr.inputs.Radio(["None", "l1", "l2"], default="None")  # ditto

    inputs = [
        gr.inputs.File(label="file 1"),
        gr.inputs.File(label="file 2", optional=True),
        input_tf_type,  # modi inputs
        input_idf_type,
        input_dl_type,
        input_norm_type,
        gr.inputs.Slider(
            minimum=1,
            maximum=20,
            step=0.1,
            default=10,
        ),
        gr.inputs.Slider(
            minimum=1,
            maximum=20,
            step=1,
            default=6,
        ),
    ]

    examples = [
        [
            "data/test_zh.txt",
            "data/test_en.txt",
            "linear",
            "None",
            "None",
            "None",
            10,
            6,
        ],
        [
            "data/test_en.txt",
            "data/test_zh.txt",
            "linear",
            "None",
            "None",
            "None",
            10,
            6,
        ],
        [
            "data/shakespeare_zh500.txt",
            "data/shakespeare_en500.txt",
            "linear",
            "None",
            "None",
            "None",
            10,
            6,
        ],
        [
            "data/shakespeare_en500.txt",
            "data/shakespeare_zh500.txt",
            "linear",
            "None",
            "None",
            "None",
            10,
            6,
        ],
        [
            "data/hlm-ch1-zh.txt",
            "data/hlm-ch1-en.txt",
            "linear",
            "None",
            "None",
            "None",
            10,
            6,
        ],
        [
            "data/hlm-ch1-en.txt",
            "data/hlm-ch1-zh.txt",
            "linear",
            "None",
            "None",
            "None",
            10,
            6,
        ],
        [
            "data/ps-cn.txt",
            "data/ps-en.txt",
            "linear",
            "None",
            "None",
            "None",
            10,
            4,
        ],
        [
            "data/test-dual.txt",
            None,
            "linear",
            "None",
            "None",
            "None",
            10,
            6,
        ],
    ]

    # modi examples setup

    outputs = ["dataframe", "plot"]
    outputs = ["plot"]
    outputs = ["dataframe", "plot"]
    out_df = gr.outputs.Dataframe(
        headers=None,
        max_rows=12,  # 20
        max_cols=None,
        overflow_row_behaviour="paginate",
        type="auto",
        label="To be aligned",
    )
    out_df_aligned = gr.outputs.Dataframe(
        headers=None,
        # max_rows=12,  # 20
        max_cols=3,
        overflow_row_behaviour="paginate",
        type="auto",
        label="aligned pairs",
    )
    out_file_dl = gr.outputs.File(
        label="Click to download csv",
    )
    out_file_dl_excel = gr.outputs.File(
        label="Click to download xlsx",
    )

    # modi outputs
    outputs = [
        out_df,
        "plot",
        out_file_dl,
        out_file_dl_excel,
        out_df_aligned,
    ]
    # outputs = ["dataframe", "plot", "plot"]  # wont work
    # outputs = ["dataframe"]
    # outputs = ["dataframe", "dataframe", ]

    # def fn(file1, file2):
    # def fn(file1, file2, min_samples, eps):
    def fn(
        file1,
        file2,
        tf_type,
        idf_type,
        dl_type,
        norm,
        eps,
        min_samples,
    ):
        # modi fn
        """Process inputs and return outputs."""
        logger.debug(" *debug* ")

        # conver "None" to None for those Radio types
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

        if not text1 or not text2:
            msg = (
                "one of the files is empty: "
                f"text1[:10]: [{text1[:10]}], text2[:10]: [{text2[:10]}]"
            )
            # return (pd.DataFrame([msg], columns=['error message']), *((None,) * 4))
            return error_msg(msg)

        lang1, _ = fastlid(text1)
        lang2, _ = fastlid(text2)

        df1 = files2df(file1, file2)

        lst1 = [elm for elm in df1.text1 if elm]
        lst2 = [elm for elm in df1.text2 if elm]
        # len1 = len(lst1)  # noqa
        # len2 = len(lst2)  # noqa

        cmat = lists2cmat(
            lst1,
            lst2,
            tf_type=tf_type,
            idf_type=idf_type,
            dl_type=dl_type,
            norm=norm,
        )

        tset = pd.DataFrame(cmat2tset(cmat))
        tset.columns = ["x", "y", "cos"]

        df_trimmed = trim_df(df1)
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

        # process lst1, lst2 to obtained df_aligned
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
        plt.switch_backend('Agg')  # to cater for Mac, thanks to WhiteFox

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
            "max along columns ('x': outliers)\n"
            "potential aligned pairs (green line)\n"
            f"({round(sum(_) / xlim, 2):.0%})"
        )

        # clustered
        # df_[_].plot.scatter("x", "y", c="cos", cmap=cmap, ax=ax1)
        # ax1.set_xlabel(xlabel)
        # ax1.set_ylabel(ylabel)
        # ax1.set_xlim(0, len1)
        # ax1.set_title(f"potential aligned pairs ({round(sum(_) / len1, 2):.0%})")
        # end of plot_cmat

        src_len, tgt_len = cmat.shape
        aset = gen_aset(pset, src_len, tgt_len)
        final_list = align_texts(aset, lst2, lst1)  # note the order

        # df_aligned = df_trimmed
        df_aligned = pd.DataFrame(final_list, columns=["text1", "text2", "likelihood"])

        # swap text1 text2
        df_aligned = df_aligned[["text2", "text1", "likelihood"]]
        df_aligned.columns = ["text1", "text2", "likelihood"]

        _ = df_aligned.to_csv(index=False)
        file_dl = Path(f"{Path(file1.name).stem[:-8]}-{Path(file2.name).stem[:-8]}.csv")
        file_dl.write_text(_, encoding="utf8")

        # file_dl.write_text(_, encoding="gb2312")  # no go

        file_dl_xlsx = Path(
            f"{Path(file1.name).stem[:-8]}-{Path(file2.name).stem[:-8]}.xlsx"
        )
        df_aligned.to_excel(file_dl_xlsx)

        # return df_trimmed, plt
        return df_trimmed, plt, file_dl, file_dl_xlsx, df_aligned
        # modi outputs

    server_port = 7860
    with socket(AF_INET, SOCK_STREAM) as sock:
        sock.settimeout(0.01)  # 10ms

        # try numb times before giving up
        numb = 5
        for _ in range(numb):
            if sock.connect_ex(("127.0.0.1", server_port)) != 0:  # port idle
                break
            server_port = server_port + randint(0, 50)
        else:
            raise SystemExit(f"Tried {numb} times to no avail, giving up...")

    # moved to userguide.rst in docs
    article = dedent(
        """
        ## NB
        *   `radiobee aligner` is a sibling of `bumblebee aligner`. To know more about these aligners, please join qq group `316287378`.
        *   Uploaded files should be in pure text format (txt, md, csv etc). `docx`, `pdf`, `srt`, `html` etc may be supported later on.
        *   Click "Clear" first for subsequent submits when uploading files.
        *   `tf_type` `idf_type` `dl_type` `norm`: Normally there is no need to touch these unless you know what you are doing.
        *   Suggested `esp` and `min_samples` values -- `esp` (minimum epsilon): 8-12, `min_samples`: 4-8.
           -   Smaller larger `esp` or `min_samples` will result in more aligned pairs but also more **false positives** (pairs
           falsely identified as candidates). On the other hand,
           larger smaller `esp` or `min_samples` values tend to miss
           'good' pairs.
        *   If you need to have a better look at the image, you can right-click on the image and select copy-image-address and open a new tab in the browser with the copied image address.
        *   `Flag`: Should something go wrong, you can click Flag to save the output and inform the developer.
        """
    ).strip()

    # "<p style='text-align: center'><a href='https://arxiv.org/abs/2112.11641' target='_blank'>JoJoGAN: One Shot Face Stylization</a>| <a href='https://github.com/mchong6/JoJoGAN' target='_blank'>Github Repo Pytorch</a></p> <center><img src='https://visitor-badge.glitch.me/badge?page_id=akhaliq_jojogan' alt='visitor badge'></center> <p style='text-align: center'>samples from repo: <img src='https://raw.githubusercontent.com/mchong6/JoJoGAN/main/teaser.jpg' alt='animation'/></p>"  # noqa

    article = dedent(
        """ <p style="text-align: center">
        <a href="https://radiobee.readthedocs.io/" target="_blank">readthedocs</a>
         or <a href="https://radiobee.readthedocs.io/en/latest/userguide-zh.html#" target="_blank">中文使用说明</a>
        </p>
        """
    ).strip()

    css_image = ".output_image, .input_image {height: 40rem !important; width: 100% !important;}"
    # css = ".output_image, .input_image {height: 20rem !important; width: 100% !important;}"
    css_input_file = (
        ".input_file {height: 9rem !important; width: 100% !important;}"
    )
    css_output_file = (
        ".output_file {height: 4rem !important; width: 100% !important;}"
    )

    logger.info("running at port %s", server_port)

    iface = gr.Interface(
        # fn=greet,
        # inputs="text",
        # fn=process_upload,
        # fn=process_2upoads,
        # inputs=["file", "file"],
        # outputs="text",
        # outputs="html",
        fn=fn,
        inputs=inputs,
        outputs=outputs,
        title="radiobee-aligner🔠",
        description="WIP showcasing a blazing fast dualtext aligner, currrently supported language pairs: en-zh/zh-en",
        article=article,
        examples=examples,
        # theme="darkgrass",
        theme="grass",
        layout="vertical",  # horizontal unaligned
        allow_flagging="auto",
        flagging_options=[
            "fatal",
            "bug",
            "brainstorm",
            "excelsior",
        ],  # "paragon"],
        css=f"{css_image} {css_input_file} {css_output_file}",
    )

    iface.launch(
        share=False,
        # share=True,
        debug=debug,
        # server_name="0.0.0.0",
        server_name="127.0.0.1",
        server_port=server_port,
        # show_tips=True,
        enable_queue=True,
        # height=150,  # 500
        width=900,  # 900
    )

_ = """

        ax = sns.heatmap(cmat, cmap="viridis_r")

        ax.invert_yaxis()
        ax.set_xlabel(fastlid(df.text1)[0])
        ax.set_xlabel(fastlid(df.text2)[0])

        # return df, plt
        return plt.gca()

https://colab.research.google.com/drive/1Gz9624VeAQLT7wlETgjOjPVURzQckXI0#scrollTo=qibtTvwecgsL colab gradio-file-inputs-upload.ipynb
    iface = gr.Interface(plot_text, "file", "image")

def is_port_in_use(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect_ex(('127.0.0.1', 7911))

---
css https://huggingface.co/spaces/nielsr/LayoutLMv2-FUNSD/blob/main/app.py#L83

css = ".output_image, .input_image {height: 40rem !important; width: 100% !important;}"
#css = "@media screen and (max-width: 600px) { .output_image, .input_image {height:20rem !important; width: 100% !important;} }"
# css = ".output_image, .input_image {height: 600px !important}"

mod = 'en2zh'
packname = packx.__name__

globals()[mod] = getattr(importlib.import_module(f"{packname}.{mod}"), mod)

"""
