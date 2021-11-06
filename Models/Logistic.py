from sklearn.linear_model import LogisticRegression
import pandas as pd
import statsmodels.api as sm
import numpy as np
from statsmodels.discrete.conditional_models import ConditionalLogit
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

def log_reg_pred(df, variables):
    # check for the distribution of the target variable:
    print(df.ret_1.value_counts() / len(df.ret_1))
    # make train- test split based on date, not random to avoid look ahead bias
    model_data = df[df.Date < "2019-10-01"]
    backtesting_data = df[df.Date >= "2019-10-01"]
    # Split the data into explanatory and dependable variables for test and train datasets
    X = model_data[variables]
    y = model_data['return']

    X_test = backtesting_data[variables]
    y_test = backtesting_data['return']
    regr = LogisticRegression()

    regr.fit(X, y)
    backtesting_data['prediction'] = regr.predict(X_test)
    print(pd.DataFrame([pd.DataFrame(backtesting_data['prediction']).value_counts(), pd.DataFrame(y_test).value_counts()],
                       index=['Prediction', 'Y_test']).T)
    print(accuracy_score(backtesting_data['prediction'], backtesting_data['return']))
    # backtesting_data['strategy'] = backtesting_data['prediction'] * backtesting_data['ret']
    # print((backtesting_data[['ret', 'strategy']]).sum().apply(np.exp))
    # backtesting_data[['ret', 'strategy']].cumsum().apply(np.exp).plot(figsize = (10, 6))
    # plt.show()
    y = (y > 0).astype(int)
    logit_model = ConditionalLogit(y.to_numpy(), X.astype(float).to_numpy(), groups=model_data['coin'].to_numpy())

    # fit logit model into the data
    result = logit_model.fit()

    # summarize the logit model
    print(result.summary())

def log_reg(df, variables):
    # check for the distribution of the target variable:
    print(df.ret_1.value_counts() / len(df.ret_1))

    X = df[variables]
    y = df['return']
    y = (y > 0).astype(int)
    logit = sm.Logit(y.to_numpy(), X.astype(float).to_numpy())
    logit_model = ConditionalLogit(y.to_numpy(), X.astype(float).to_numpy(), groups=df['coin'].to_numpy())

    # fit logit model into the data
    result = logit_model.fit()
    result2 = logit.fit()
    # summarize the logit model
    print(result.summary())
    print(result2.summary())


sentiment_telegram = pd.read_csv(
    '/Users/torbenleowald/Documents/Python Finance/python_trading_bot/Dataset/telegram.csv')
merged = pd.read_csv('/Users/torbenleowald/Documents/Python Finance/python_trading_bot/Dataset/merged.csv')

print(merged.info())
variables = ['average', 'ret_log', 'score', 'std', 'google_trend', '1', '2', '3', '4', 'vola', 'ret_google_trend', 'vola_score']
log_reg(merged, variables)
