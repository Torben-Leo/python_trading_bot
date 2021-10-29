
from sklearn.linear_model import LogisticRegression
from Dataset.Data_download import urls_to_df_from_url
import pandas as pd
import statsmodels.api as sm
import numpy as np
from statsmodels.discrete.conditional_models import ConditionalLogit




def log_reg(df):
    # check for distribution of target variable:
    print(sentiment_reddit_df.ret_1.value_counts() / len(sentiment_reddit_df.ret_1))
    # make train- test split based on date, not random
    model_data = df[df.start < "2019-10-01"]
    backtesting_data = df[df.start >= "2019-10-01"]

    print(model_data.coin.value_counts())
    # Split the data into explanatory and dependable variables for test and train datasets
    X = model_data[['average', 'ret']]
    y = model_data['return']

    X_test = backtesting_data[['average', 'ret']]
    y_test = backtesting_data['return']
    regr = LogisticRegression()

    regr.fit(X, y)
    y_pred = regr.predict(X_test)
    print(pd.DataFrame([pd.DataFrame(y_pred).value_counts(), pd.DataFrame(y_test).value_counts()],
                 index=['Prediction', 'Y_test']).T)
    logit_model = ConditionalLogit(y.to_numpy(), X.astype(float).to_numpy(), groups = model_data['coin'].to_numpy())

    # fit logit model into the data
    result = logit_model.fit()

    # summarize the logit model
    print(result.summary())





sentiment_reddit_df = pd.read_csv('reddit.csv')
sentiment_telegram = pd.read_csv('telegram.csv')
print(sentiment_reddit_df.describe())
log_reg(sentiment_reddit_df)