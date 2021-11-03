import yfinance as yf
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller

def downlaod_tickers(ticker):
    start = dt.datetime(2017, 6,1)
    end = dt.datetime.now()
    df = yf.download(ticker, period='max', threads=True)
    plt.figure(figsize=(20, 10))
    df['Adj Close'].plot()
    plt.ylabel("Daily prices of Bitcoin in USD")
    plt.show()
    df['Return'] = df['Adj Close'].pct_change()
    df = df[1:]
    df.head()
    df['Return'].plot()
    plt.ylabel("Daily returns of Bitcoin in USD")
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 3))
    plot_acf(df['Adj Close'], ax=ax, lags=500);

    fig, ax = plt.subplots(figsize=(8, 3))
    plot_pacf(df['Adj Close'], ax=ax, lags=500);

    fig, ax = plt.subplots(figsize=(8, 3))
    plot_acf(df['Return'], ax=ax, lags=100);

    fig, ax = plt.subplots(figsize=(8, 3))
    plot_pacf(df['Return'], ax=ax, lags=100);
    return df




def test_stationarity(timeseries, name):
    # Determining rolling statistics
    rolmean = timeseries.rolling(14).mean()
    rolstd = timeseries.rolling(14).std()

    # Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation' + name)
    plt.show(block=False)

    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:' + name)
    timeseries = timeseries.iloc[:].values
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)

urls = ['ada', 'algo', 'atom','bat', 'bch', 'bnb', 'btc', 'cvc', 'dash', 'dnt', 'doge', 'eos', 'gnt', 'knc',
        'link', 'loom', 'ltc', 'mana', 'mkr', 'neo', 'rep', 'trx', 'xem', 'xlm', 'xrp', 'xtz', 'zec', 'zrx']
tickers = ['BTC-USD']
''', 'ETH-USD', 'XRP-USD', 'ADA-USD','BNB-USD', 'HEX-USD', 'SOL1-USD','DOGE-USD']'''
for url in urls:
    df = downlaod_tickers(url.capitalize()+'-USD')
    test_stationarity(df['Adj Close'], url)
    test_stationarity(df['Return'], url)


