import statsmodels.api as sm
import pandas as pd
from stargazer.stargazer import Stargazer
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from IPython.core.display import HTML

import matplotlib.pyplot as plt
def lin_reg(df, variables):
    # check for the distribution of the target variable:
    print(df.ret_1.value_counts() / len(df.ret_1))
    X = sm.add_constant(df[variables])
    y = df['ret_1']
    regr = sm.OLS(y, X)
    res = regr.fit(cov_type='HC1')
    # summarize the multiple linear regression model
    print(res.summary())
    return res

sentiment_telegram = pd.read_csv(
    '/Users/torbenleowald/Documents/Python Finance/python_trading_bot/Dataset/telegram.csv')
merged = pd.read_csv('/Users/torbenleowald/Documents/Python Finance/python_trading_bot/Dataset/merged.csv')
variables5 = ['ret', 'score',  'google_trend_change', 'volumeUSD', 'vola', 'ret_google_trend', 'vola_score']
lin_regression = lin_reg(merged, variables5)
# table creation
stargazer = Stargazer([lin_regression])
print(HTML(stargazer.render_html()))