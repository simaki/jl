%reload_ext autoreload
%autoreload

import glob
import os
import pathlib
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
from tqdm import tqdm

matplotlib.rcParams["figure.figsize"] = (24, 8)
matplotlib.rcParams["figure.dpi"] = 150
seaborn.set_style("whitegrid")

import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.random.seed(42)
_ = torch.manual_seed(42)
