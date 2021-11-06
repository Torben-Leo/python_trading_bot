from prophet import Prophet
from prophet.plot import plot_plotly
from prophet.plot import add_changepoints_to_plot
import pandas as pd
#from Dataset.timeseries_data import downlaod_tickers
import matplotlib.pyplot as plt
from Helper.yFinance import download_ticker
import datetime as dt

def prediction(data, ticker):
    # reset index for Date column
    df = data.reset_index()
    # create a dataframe
    train_dataset = pd.DataFrame()

    train_dataset['ds'] = df['Date']
    train_dataset['y'] = df['ret']
    train_dataset.head(2)
    # create a Prophet instance with default values to fit the dataset
    prophet_basic = Prophet()
    prophet_basic.fit(train_dataset)
    # create a dataframe with ds (i.e., datetime stamp) that has the time series of dates we need for prediction
    # periods specify the number of days to extend into the future
    future = prophet_basic.make_future_dataframe(periods=100)
    #future.tail()
    # forecast BTC prices
    forecast = prophet_basic.predict(future)
    # plot predicted BTC prices
    #fig1 = prophet_basic.plot(forecast)

    # plot the trend and seasonality
    #fig1 = prophet_basic.plot_components(forecast)
    # identify changepoints (i.e., datetime points when the time series exprience abrupt changes)

    fig = prophet_basic.plot(forecast)
    plt.title(ticker)
    a = add_changepoints_to_plot(fig.gca(), prophet_basic, forecast)
    #print(prophet_basic.changepoints)
    # adjust trend sensitivity with "changepoint_prior_scale" parameter
    # default value is 0.05. Lower value, less flexible trend, and vice versa

    pro_change = Prophet(changepoint_prior_scale=0.15)
    forecast = pro_change.fit(train_dataset).predict(future)
    fig = pro_change.plot(forecast)
    a = add_changepoints_to_plot(fig.gca(), prophet_basic, forecast)
    plt.title(ticker)
    plt.show()
    return forecast


#tickers = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'ADA-USD','BNB-USD', 'HEX-USD','DOGE-USD']
tickers = ['ada', 'atom','bat', 'bch', 'bnb', 'btc', 'cvc', 'dai', 'dash', 'dnt', 'doge', 'eos', 'gnt', 'knc',
        'link', 'loom', 'ltc', 'mana', 'mkr', 'neo', 'rep', 'trx', 'xem', 'xlm', 'xrp', 'xtz', 'zec', 'zrx']
timeseries = pd.DataFrame()
for ticker in tickers:
    df = download_ticker(ticker.capitalize() + '-USD')
    if df.size > 5:
        forecast = prediction(df, ticker)
        forecast['coin'] = ticker
        timeseries = timeseries.append(forecast[forecast.ds > dt.datetime(2019, 9, 29)][['ds', 'coin', 'yhat']])
        timeseries['yhat'] = (timeseries['yhat']> 0).astype(int)

timeseries.info()
timeseries.to_csv('timeseries.csv')