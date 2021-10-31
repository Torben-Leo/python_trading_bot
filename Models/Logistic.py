from sklearn.linear_model import LogisticRegression
import pandas as pd
import statsmodels.api as sm
import numpy as np
from statsmodels.discrete.conditional_models import ConditionalLogit
import warnings
warnings.filterwarnings('ignore')

def log_reg(df, variables):
    # check for distribution of target variable:
    print(df.ret_1.value_counts() / len(df.ret_1))
    # make train- test split based on date, not random
    model_data = df[df.Date < "2019-10-01"]
    backtesting_data = df[df.Date >= "2019-10-01"]

    print(model_data.coin.value_counts())
    # Split the data into explanatory and dependable variables for test and train datasets
    X = model_data[variables]
    y = model_data['return']

    X_test = backtesting_data[variables]
    y_test = backtesting_data['return']
    regr = LogisticRegression()

    regr.fit(X, y)
    y_pred = regr.predict(X_test)
    print(pd.DataFrame([pd.DataFrame(y_pred).value_counts(), pd.DataFrame(y_test).value_counts()],
                       index=['Prediction', 'Y_test']).T)
    logit_model = ConditionalLogit(y.to_numpy(), X.astype(float).to_numpy(), groups=model_data['coin'].to_numpy())

    # fit logit model into the data
    result = logit_model.fit()

    # summarize the logit model
    print(result.summary())


sentiment_telegram = pd.read_csv(
    '/Users/torbenleowald/Documents/Python Finance/python_trading_bot/Dataset/telegram.csv')
merged = pd.read_csv('/Users/torbenleowald/Documents/Python Finance/python_trading_bot/Dataset/merged.csv')
print(merged.info())
variables = ['average', 'ret', 'google_trend', 'score', '1', '2', '3', '4', 'vola']
log_reg(merged, variables)
