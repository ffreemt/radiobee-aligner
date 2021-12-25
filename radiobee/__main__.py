"""Run interactively."""
from typing import Tuple  # , Optional

import joblib
from random import randint
from textwrap import dedent
from itertools import zip_longest
from sklearn.cluster import DBSCAN

from socket import socket, AF_INET, SOCK_STREAM
import signal
from varname import nameof
from logzero import logger

# import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# from tabulate import tabulate
from fastlid import fastlid

import gradio as gr
from radiobee.process_upload import process_upload
from radiobee.files2df import files2df
from radiobee.file2text import file2text
from radiobee.lists2cmat import lists2cmat

# from radiobee.plot_df import plot_df
from radiobee.cmat2tset import cmat2tset

sns.set()
sns.set_style("darkgrid")
fastlid.set_languages = ["en", "zh"]

signal.signal(signal.SIGINT, signal.SIG_DFL)
print("Press Ctrl+C to quit\n")


def savelzma(obj, fileloc: str = None):
    if fileloc is None:
        fileloc = nameof(obj)  # this wont work
    joblib.dump(obj, f"data/{fileloc}.lzma")


def greet(input):
    """Greet yo."""
    return f"'Sup yo! (your input: {input})"


def upfile1(file1, file2=None) -> Tuple[str, str]:
    """Upload file1, file2."""
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
    # import logzero
    # logzero.loglevel(10)
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
    inputs = [
        gr.inputs.File(label="file 1"),
        gr.inputs.File(label="file 2", optional=True),
        gr.inputs.Slider(
            minimum=1,
            maximum=20,
            step=1,
            default=6,
            # label="suggested min_samples value: 4-8",
        ),
        gr.inputs.Slider(
            minimum=1,
            maximum=20,
            step=0.1,
            default=2,
            # label="suggested esp value: 1.7-3",
        ),
    ]

    examples = [
        ["data/test_zh.txt", "data/test_en.txt", 6, 10, ],
        ["data/test_en.txt", "data/test_zh.txt", 6, 10, ],
        ["data/shakespeare_zh500.txt", "data/shakespeare_en500.txt", 6, 10, ],
        ["data/shakespeare_en500.txt", "data/shakespeare_zh500.txt", 6, 10, ],
        ["data/hlm-ch1-zh.txt", "data/hlm-ch1-en.txt", 6, 10, ],
        ["data/hlm-ch1-en.txt", "data/hlm-ch1-zh.txt", 6, 10, ],
    ]
    outputs = ["dataframe", "plot"]
    outputs = ["plot"]
    outputs = ["dataframe", "plot"]
    out1 = gr.outputs.Dataframe(
        headers=None,
        max_rows=12,  # 20
        max_cols=None,
        overflow_row_behaviour="paginate",
        type="auto",
        label="To be aligned",
    )
    outputs = [
        out1,
        "plot",
    ]
    # outputs = ["dataframe", "plot", "plot"]  # wont work
    # outputs = ["dataframe"]
    # outputs = ["dataframe", "dataframe", ]

    # def fn(file1, file2):
    def fn(file1, file2, min_samples, eps):
        """Process inputs."""
        logger.debug(" *debug* ")

        # logger.info("file1: *%s*, file2: *%s*", file1, file2)
        logger.info("file1.name: *%s*, file2.name: *%s*", file1.name, file2.name)

        # bypass if file1 or file2 is str input
        if not (isinstance(file1, str) or isinstance(file2, str)):
            text1 = file2text(file1)
            text2 = file2text(file2)
            lang1, _ = fastlid(text1)
            lang2, _ = fastlid(text2)

            df1 = files2df(file1, file2)

            lst1 = [elm for elm in df1.text1 if elm]
            lst2 = [elm for elm in df1.text2 if elm]
            len1 = len(lst1)
            len2 = len(lst2)

            # this wont work
            # for obj in [text1, text2, df1, lst1, lst2, ]:
            # savelzma(text1) wont work

            # for debugging
            # joblib.dump(text1, f"data/{nameof(text1)}.lzma")
            # joblib.dump(text2, f"data/{nameof(text2)}.lzma")
            # joblib.dump(df1, f"data/{nameof(df1)}.lzma")
            # joblib.dump(lst1, f"data/{nameof(lst1)}.lzma")
            # joblib.dump(lst2, f"data/{nameof(lst2)}.lzma")

            cmat = lists2cmat(lst1, lst2)

            tset = pd.DataFrame(cmat2tset(cmat))
            tset.columns = ["x", "y", "cos"]

            # for debugging, logger.debug logger.info dont show up
            # print("lst1: %s" % lst1)
            # print("lst2: %s" % lst2)
            # print("cmat: %s" % cmat)
            # print("tset: %s" % tset)

            logger.debug("lst1: %s", lst1)
            logger.debug("lst2: %s", lst2)
            logger.debug("cmat: %s", cmat)
            logger.debug("tset: %s", tset)

            # plt0 = plot_df(pd.DataFrame(cmat))
            df_ = tset

            # moved to inputs
            # min_samples: int = 6
            # eps: float = 10

            # ylim: Optional[int] = None
            xlabel: str = lang1
            ylabel: str = lang2

            sns.set()
            sns.set_style("darkgrid")

            # fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.69, 8.27))
            # fig, ([ax2, ax0], [ax1, ax3]) = plt.subplots(2, 2, figsize=(11.69, 8.27))
            # fig, (ax2, ax0, ax1) = plt.subplots(3)
            # fig, (ax2, ax0, ax1) = plt.subplots(3, figsize=(11.69, 8.27))
            # fig, (ax2, ax0, ax1) = plt.subplots(1, 3, figsize=(36.69, 8.27))
            # fig, (ax2, ax0, ax1) = plt.subplots(1, 3, figsize=(66.69, 22.27))
            # fig, (ax2, ax0, ax1) = plt.subplots(1, 3)
            # fig.subplots_adjust(hspace=.4)

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
            _x = DBSCAN(min_samples=min_samples, eps=eps).fit(df_).labels_ < 0

            df_.plot.scatter("x", "y", c="cos", cmap=cmap, ax=ax0)

            # clustered
            df_[_].plot.scatter("x", "y", c="cos", cmap=cmap, ax=ax1)

            # outliers
            df_[_x].plot.scatter("x", "y", c="r", marker="x", alpha=0.6, ax=ax0)

            # ax0.set_xlabel("")
            # ax0.set_ylabel("zh")
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

            # return df, plot_df(pd.DataFrame(cmat))
            # tset.plot.scatter("x", "y", c="cos", cmap="viridis_r")
        else:
            fig, ax1 = plt.subplots()
            df1 = pd.DataFrame(
                [
                    [5.1, 3.5, 0],
                    [4.9, 3.0, 0],
                    [7.0, 3.2, 1],
                    [6.4, 3.2, 1],
                    [5.9, 3.0, 2],
                ],
                columns=["length", "width", "species"],
            )
            df1.plot.scatter(x="length", y="width", c="DarkBlue", ax=ax1)
            # plt_heatmap = plt

        # plt.scatter(df.length, df.width)  # gradio eturn plt.gcf() or plt

        # return df, plt
        # return plt
        # return df, df
        # return df1.iloc[:10, :], plt

        # pd.concat([df0, pd.DataFrame([[".", ".", "..."]], columns=df0.columns)], ignore_index=1)
        # pd.concat([df0.iloc[:2, :], pd.DataFrame([[".", ".", "..."]], columns=df0.columns),  df0.iloc[-1:, :]], ignore_index=1)

        # _ = pd.concat([df1.iloc[:4, :], pd.DataFrame([["...", "...", "...", ]], columns=df1.columns), df1.iloc[-2:, :]], ignore_index=True)
        # _ = pd.concat([df.iloc[:2, :], pd.DataFrame([[".", ".", "..."]], columns=df.columns),  df.iloc[-1:, :]], ignore_index=1)

        _ = pd.concat(
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

        return _, plt
        # return _, plt

    # """

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

    article = dedent(
        """
        ## NB
        *   Click "Clear" first for subsequent submits when uploading files.
        *   Suggested values : min_samples: 4-8, esp (minimum epsilon): 8-12. 
           -   Smaller min_samples or larger esp will result in more aligned pairs but also more **false positives** (pairs falsly identified as candidates). On the other hand, larger min_samples or smaller esp values tend to miss 'good' pairs.
    """
    )
    css = ".output_image, .input_image {height: 40rem !important; width: 100% !important;}"
    # css = ".output_image, .input_image {height: 20rem !important; width: 100% !important;}"
    css_file = (
        ".input_file, .output_file {height: 9rem !important; width: 100% !important;}"
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
        description="showcasing a blazing fast dualtext aligner, currrently supported language pairs: en-zh/zh-en",
        article=article,
        examples=examples,
        # theme="darkgrass",
        layout="vertical",  # horizontal unaligned
        # height=150,  # 500
        width=900,  # 900
        allow_flagging=False,
        flagging_options=["fatal", "bug", "brainstorm", "excelsior", "paragon"],
        css=f"{css} {css_file}",
    )

    iface.launch(
        # share=False,
        share=True,
        debug=True,
        server_name="0.0.0.0",
        server_port=server_port,
        # show_tips=True,
        enable_queue=True,
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
