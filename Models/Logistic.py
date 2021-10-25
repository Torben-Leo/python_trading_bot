
from sklearn.linear_model import LogisticRegression
from Dataset.Data_download import urls_to_df_from_url
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols, logit




def log_reg(df):
    # check for distribution of target variable:
    print(sentiment_df.ret_1.value_counts() / len(sentiment_df.ret_1))
    # make train- test split based on date, not random
    model_data = df[df.index < "2019-10-01"]
    backtesting_data = df[df.index >= "2019-10-01"]

    # Split the data into explanatory and dependable variables for test and train datasets
    X = model_data[['score', 'volumeUSD', 'vola']]
    y = model_data['ret_1']

    X_test = backtesting_data[['score', 'volumeUSD', 'vola']]
    y_test = backtesting_data['ret_1']
    regr = LogisticRegression()
    regr.fit(X, y)
    y_pred = regr.predict(X_test)
    print(pd.DataFrame([pd.DataFrame(y_pred).value_counts(), pd.DataFrame(y_test).value_counts()],
                 index=['Prediction', 'Y_test']).T)
    logit_model = sm.Logit(y, X.astype(float), missing='drop')

    # fit logit model into the data
    result = logit_model.fit()

    # summarize the logit model
    print(result.summary2())


urls = ['ada', 'algo', 'atom','bat', 'bch', 'bnb', 'btc', 'cvc', 'dai', 'dash', 'dnt', 'doge', 'eos', 'gnt', 'knc',
        'link', 'loom', 'ltc', 'mana', 'mkr', 'neo', 'rep', 'trx', 'xem', 'xlm', 'xrp', 'xtz', 'zec', 'zrx']
sentiment_df = urls_to_df_from_url(urls).dropna()
print(sentiment_df.describe())
log_reg(sentiment_df)