"""Load model_s."""
# pylint: disable=invalid-name

from pathlib import Path

import joblib
from huggingface_hub import hf_hub_url, cached_download  # hf_hub_download,
from alive_progress import alive_bar
from logzero import logger


def load_model_s():
    """Load local model_s if present, else fetch from hf.co."""
    file_loc = "radiobee/model_s"
    if Path(file_loc).exists():
        # raise Exception(f"File {file_loc} does not exist.")

        with alive_bar(1, title=" Loading model_s, takes ~30 secs ...", length=3) as progress_bar:
            model = joblib.load(file_loc)

            # model_s = pickle.load(open(file_loc, "rb"))
            progress_bar()  # pylint: disable=not-callable

        return model

    logger.info(
        "Fetching and caching model_s from huggingface.co... "
        "The first time may take a while depending on your net."
    )
    with alive_bar(1, title=" Subsequent loading takes ~20 secs ...", length=3) as progress_bar:
        model = joblib.load(cached_download(hf_hub_url("mikeee/model_s", "model_s")))
        progress_bar()  # pylint: disable=not-callable

    return model


model_s = load_model_s()
