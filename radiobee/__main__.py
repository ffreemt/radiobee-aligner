"""Run interactively."""
# pylint: disable=invalid-name, too-many-arguments, unused-argument, redefined-builtin, unused-import, wrong-import-position, too-many-locals, too-many-statements
from typing import Any, Tuple, Optional, Union  # noqa

import sys
from pathlib import Path  # noqa
import subprocess as sp
import shlex
import platform
import signal
from random import randint
from textwrap import dedent
from itertools import zip_longest

# import socket
from socket import socket, AF_INET, SOCK_STREAM

from sklearn.cluster import DBSCAN  # noqa

import joblib
from varname import nameof
from icecream import install as ic_install, ic
import logzero
from logzero import logger

# import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt  # noqa

# from tabulate import tabulate
from fastlid import fastlid

# for embeddable python
if "." not in sys.path:
    sys.path.insert(0, ".")

import gradio as gr

# from radiobee.error_msg import error_msg
from radiobee.process_upload import process_upload
from radiobee.gradiobee import gradiobee

ic_install()
ic.configureOutput(
    includeContext=True,
    outputFunction=logger.info,
)
ic.enable()
# ic.disenable()  # to turn off

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


if __name__ == "__main__":
    debug = True
    # debug = False

    uname = platform.uname()

    # match = re.search(r'[a-z\d]{10,}', gethostname())
    # hf spaces release: '4.14.248-189.473.amzn2.x86_64'
    # match = re.search(r'[a-z\d]{10,}', node)
    # if match and node.system.lower() in ["linux"]:
    if "amzn2" in uname.release:
        # likely hf spaces
        server_name = "0.0.0.0"
        debug = False
        debug = True
<<<<<<< HEAD
        share = True

        # set UTC+8, probably wont work in hf spaces, no permission
        try:
            sp.check_output(shlex.split("ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime"))
        except Exception as exc:
            logger.error(" set timezonef failed: %s", exc)
=======
        share = False
>>>>>>> refs/remotes/origin/main
    else:
        server_name = "127.0.0.1"
        share = False

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

    # modi inputs 1, definitions
    sent_ali_algo = gr.inputs.Radio(["None", "fast", "slow"], default="None")

    inputs = [  # tot. 9, meed to modify input of gradio & examples
        gr.inputs.File(label="file 1"),
        gr.inputs.File(label="file 2", optional=True),
        input_tf_type,  # modi inputs 2
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
        sent_ali_algo,
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
            "None",
        ],
        [
            "data/test_zh.txt",
            "data/test_en.txt",
            "linear",
            "None",
            "None",
            "None",
            10,
            6,
            "fast",
        ],
        [
            "data/test_zh.txt",
            "data/test_en.txt",
            "linear",
            "None",
            "None",
            "None",
            10,
            6,
            "slow",
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
            "None",
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
            "None",
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
            "None",
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
            "None",
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
            "None",
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
            "None",
        ],
        [
            "data/test-dual.txt",
            "data/empty.txt",
            "linear",
            "None",
            "None",
            "None",
            10,
            6,
            "None",
        ],
        [
            "data/英译中国现代散文选1(汉外对照丛书).txt",
            "data/empty.txt",
            "linear",
            "None",
            "None",
            "None",
            10,
            6,
            "None",
        ],
        [
            "data/test-zh-ja.txt",
            "data/empty.txt",
            "linear",
            "None",
            "None",
            "None",
            10,
            6,
            "None",
        ],
        [
            "data/xiyouji-ch1-zh.txt",
            "data/xiyouji-ch1-de.txt",
            "linear",
            "None",
            "None",
            "None",
            10,
            6,
            "None",
        ],
        [
            "data/demian-hesse-de.txt",
            "data/demian-hesse-en.txt",
            "linear",
            "None",
            "None",
            "None",
            10,
            6,
            "None",
        ],
        [
            "data/catcher-in-the-rye-shixianrong-zh.txt",
            "data/catcher-in-the-rye-boll-de.txt",
            "linear",
            "None",
            "None",
            "None",
            10,
            6,
            "None",
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
    out_sents_dl = gr.outputs.File(
        label="Click to download sents csv",
    )
    out_sents_dl_excel = gr.outputs.File(
        label="Click to download sents xlsx",
    )

    # modi outputs 1, definitions

    # modi outputs 2, need to modify gradio error_msg
    outputs = [  # tot. 8
        out_df,
        gr.outputs.Image(label="plot"),
        out_file_dl,
        out_file_dl_excel,
        out_sents_dl,
        out_sents_dl_excel,
        out_df_aligned,
        gr.outputs.HTML(),
    ]
    # outputs = ["dataframe", "plot", "plot"]  # wont work
    # outputs = ["dataframe"]
    # outputs = ["dataframe", "dataframe", ]

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

    description = "WIP showcasing a blazing fast dualtext aligner, currrently supported language pairs: en-zh/zh-en for fast-track, other language pairs are handled by slow-track"

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

    article = dedent(
        """ <p style="text-align: center">readiobee docs:
        <a href="https://radiobee.readthedocs.io/" target="_blank">readthedocs</a>
         or in Chinese but rather short <a href="https://radiobee.readthedocs.io/en/latest/userguide-zh.html#" target="_blank">中文使用说明</a>
        </p>
        """
    ).strip()

    css_image = ".output_image, .input_image {height: 40rem !important; width: 100% !important;}"

    # css = ".output_image, .input_image {height: 20rem !important; width: 100% !important;}"

    css_input_file = ".input_file  {height: 8rem !important; width: 100% !important;}"

    css_output_file = ".output_file  {height: 4rem !important; width: 100% !important;}"

    logger.info("running at port %s", server_port)

    _ = """
    inputs = ...
    outputs = ...
    # properly
    # define gradiobee to take inputs and spil out outputs

    iface = gr.Interface(
        fn=gradiobee,
        inputs,
        outputs,
    )
    # """
    iface = gr.Interface(
        # fn=greet,
        # inputs="text",
        # fn=process_upload,
        # fn=process_2upoads,
        # inputs=["file", "file"],
        # outputs="text",
        # outputs="html",
        # fn=fn,
        fn=gradiobee,
        inputs=inputs,
        outputs=outputs,
        title="radiobee-aligner🔠",
        description=description,
        article=article,
        examples=examples,
        examples_per_page=4,
        # theme="darkgrass",
        theme="grass",
        layout="vertical",  # horizontal unaligned
        allow_flagging="never",  # "auto" "manual"
        flagging_options=[
            "fatal",
            "bug",
            "brainstorm",
            "excelsior",
        ],  # "paragon"],
        css=f"{css_image} {css_input_file} {css_output_file}",
        # enable_queue=True,
    )

    iface.launch(
        share=share,
        debug=debug,
        show_error=True,
        server_name=server_name,
        # server_name="127.0.0.1",
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
