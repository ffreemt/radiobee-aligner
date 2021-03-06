"""Convert two iesl to pandas.DataFrame."""
from itertools import zip_longest
# import tempfile
import pandas as pd
from radiobee.process_upload import process_upload


def files2df(file1, file2):
    """Convert two files to pd.DataFrame."""
    text1 = [_.strip() for _ in process_upload(file1).splitlines() if _.strip()]

    # if file2 is tempfile._TemporaryFileWrapper:
    try:
        filename = file2.name
    except AttributeError:
        filename = ""
    if filename:
        text2 = [_.strip() for _ in process_upload(file2).splitlines() if _.strip()]
    else:
        text2 = [""]

    text1, text2 = zip(*zip_longest(text1, text2, fillvalue=""))

    df = pd.DataFrame({"text1": text1, "text2": text2})

    return df


_ = """
    # return tabulate(df)
    # return tabulate(df, tablefmt="grid")
    # return tabulate(df, tablefmt='html')
# """
