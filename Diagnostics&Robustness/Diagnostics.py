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

merged = pd.read_csv('/Users/torbenleowald/Documents/Python Finance/python_trading_bot/Dataset/merged.csv')[['score',
                                                                                                             'average',
                                                                                                             'google_trend', 'ret']]


def vif(X):
    vif = pd.DataFrame()
    vif["Variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print(vif)


vif(merged[['score', 'average', 'ret', 'google_trend']])

