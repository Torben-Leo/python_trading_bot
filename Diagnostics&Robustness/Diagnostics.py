# general regression diagnostic tests
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

import statsmodels.api as sm
import statsmodels.stats.outliers_influence as oinf
from statsmodels.stats.outliers_influence import variance_inflation_factor