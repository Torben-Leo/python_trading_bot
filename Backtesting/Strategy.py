from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import seaborn as sns
from Dataset.Data_download import urls_to_df_from_url
import matplotlib.pyplot as plt


def pred(data, variables):
    model_data = data[data.start < "2019-10-01"]
    backtesting_data = data[data.start >= "2019-10-01"]
    train_y = model_data.filter(['return'])
    train_X = model_data[variables]
    print(model_data['return'].value_counts())
    model = RandomForestClassifier(max_depth=4, n_estimators=50, min_samples_split=7, random_state=4321)
    model = model.fit(train_X, train_y)
    prediction_backtest = model.predict(backtesting_data[variables])
    # add the predictions to the original data and keep predictions, date & RP_ENTITY_ID as columns
    backtesting_data['Prediction_News'] = prediction_backtest
    backtesting_prediction = backtesting_data[['start', 'coin', 'Prediction_News', 'open', 'ret', 'ret_1', 'score', 'average']]
    print(accuracy_score(backtesting_data["return"], prediction_backtest))
    backtesting_prediction.Prediction_News.value_counts()
    print(confusion_matrix(backtesting_data["return"], prediction_backtest))
    return backtesting_prediction


def sma(stock_data):
    for coin in stock_data.coin.unique():
        ### for every ticker SMA21 & SMA252 are calculated
        stock_data.loc[stock_data['coin'] == coin, 'SMA7'] = stock_data[
            stock_data['coin'] == coin]['open'].rolling(window=7).mean()
        stock_data.loc[stock_data['coin'] == coin, 'SMA30'] = stock_data[
            stock_data['coin'] == coin]['open'].rolling(window=30).mean()

        ### when SMA21 is bigger than SMA252, we go long, short otherwise
        stock_data.loc[stock_data['coin'] == coin, 'PositionSMA'] = np.where(
            stock_data.loc[stock_data['coin'] == coin, 'SMA7'] > stock_data.loc[
                stock_data['coin'] == coin, 'SMA30'], 1, -1)

        ### the return of this strategy is the return times the position of the day before and stored in StrategySMA
        stock_data.loc[stock_data['coin'] == coin, 'StrategySMA'] = \
            (stock_data.loc[stock_data['coin'] == coin, 'PositionSMA'].shift(1) * stock_data
             .loc[stock_data['coin'] == coin, 'ret'])
    stock_data = stock_data[stock_data.SMA30.notna()]


def sentiment_strategy(data):
    data['Prediction_News'] = data['Prediction_News'].fillna(0)
    ### position is long if there was a positive prediction, zero otherwise
    data['Position_Strategy1'] = data['Prediction_News']
    data['Strategy1'] = np.nan
    for coin in data['coin'].unique():
        data.loc[data['coin'] == coin, 'Strategy1'] = (
                data.loc[data['coin'] == coin, 'Position_Strategy1'].shift(1) *
                data.loc[data['coin'] == coin, 'ret'])
    '''
    # one can see that the times where the prediction predicted positive even though the return was negative decreases with
    # increasing absolute returns
    for i in range(0, 20):
        print(len(data[(data["Strategy1"] > i)]) / (
                len(data[(data["Strategy1"] < -i)]) + len(data[(data["Strategy1"] > i)])))
    '''
def results(data):
    ### first the plot data is created
    d = {}
    for i in range(0, 15):
        d.update({i: [len(data[(data["Strategy1"] > i)]),
                      len(data[(data["Strategy1"] < -i)])]})

    d = pd.DataFrame.from_dict(d, orient='index')
    d = d.reset_index()
    d = d.rename(columns={0: "wins", 1: "losses", "index": "Absolute Return"})
    plot = d.melt(id_vars="Absolute Return", var_name="Performance", value_name="count")
    # then we plot the number of positive vs negative returns for the absolute returns from bigger than 0 to bigger than 14
    sns.set_style("darkgrid")
    g = sns.FacetGrid(plot, col="Absolute Return", col_wrap=5, sharey=False, sharex=False)
    g.map(sns.barplot, 'Performance', "count", palette="Greys_r")
    plt.show()



sentiment_df = pd.read_csv('reddit.csv')
predictions = pred(sentiment_df, ['score', 'average', 'ret'])
sma(predictions)
sentiment_strategy(predictions)
predictions.head()
predictions.info()
#results(predictions)

