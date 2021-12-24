import joblib
from importlib import import_module
import radiobee

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()
sns.set_style("darkgrid")
plt.ion()

for mod in ["lists2cmat", ]:
    globals()[mod] = getattr(import_module(f"{radiobee.__name__}.{mod}"), mod)

text1 = joblib.load("data/text1.lzma")
text2 = joblib.load("data/text2.lzma")
df1 = joblib.load("data/df1.lzma")
lst1 = joblib.load("data/lst1.lzma")
lst2 = joblib.load("data/lst2.lzma")

cmat = lists2cmat(lst1, lst2)
sns.heatmap(cmat, cmap="viridis_r").invert_yaxis()