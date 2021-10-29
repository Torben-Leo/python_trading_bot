import datetime

import numpy as np
import yfinance as yf
from Strategy import predictions as backtesting
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt


def long_coins():
    ### for the ids in the small dataset the returns of strategy1 are calculated and compared to the market

    ### first the performance of every coin in general is calculated (the result of a buy and hold strategy)

    performance_market = {
        "coin": np.empty([0, 5]),
        "Cumulated_Return": np.empty([0, 5]),
        "Risk": np.empty([0, 5]),
        "Sharpe-Ratio": np.empty([0, 5]),
    }

    for i in range(0, len(backtesting['coin'].unique())):
        coin = backtesting['coin'].unique()[i]
        performance_market["coin"] = np.append(performance_market["coin"], coin)

        ### calculating the sum of the returns (hier könnte man auch einfach die Summe nehmen)
        cumulated_return = (backtesting.loc[backtesting['coin'] == coin, 'ret'].mean() *
                            len(backtesting[backtesting['coin'] == coin]))

        performance_market["Cumulated_Return"] = np.append(performance_market["Cumulated_Return"], cumulated_return)

        risk = (backtesting.loc[backtesting['coin'] == coin, 'ret'].std() *
                len(backtesting[backtesting['coin'] == coin]))

        performance_market["Risk"] = np.append(performance_market["Risk"], risk)

        sharpe_ratio = cumulated_return / risk

        performance_market["Sharpe-Ratio"] = np.append(performance_market["Sharpe-Ratio"], sharpe_ratio)

    performance_market = pd.DataFrame.from_dict(performance_market).transpose()
    print(performance_market)


def sentiment_strategy():
    performance = {
        "coin": np.empty([0, 5]),
        "Cumulated_Return": np.empty([0, 5]),
        "Risk": np.empty([0, 5]),
        "Sharpe-Ratio": np.empty([0, 5]),
    }

    for i in range(0, len(backtesting['coin'].unique())):
        coin = backtesting['coin'].unique()[i]
        performance["coin"] = np.append(performance["coin"], coin)
        cumulated_return = (backtesting.loc[(backtesting['coin'] == coin) & (backtesting[
                                                                                     'Strategy1'] != 0),
                                            'Strategy1'].mean() * len(backtesting[(backtesting[
                                                                                       'coin'] == coin) & (
                                                                                          backtesting[
                                                                                              'Strategy1'] != 0)]))

        performance["Cumulated_Return"] = np.append(performance["Cumulated_Return"], cumulated_return)

        annualized_risk = (backtesting.loc[(backtesting['coin'] == coin) & (backtesting[
                                                                                    'Strategy1'] != 0), 'Strategy1'].std() *
                           len(backtesting[
                                   (backtesting['coin'] == coin) & (backtesting['Strategy1'] != 0)]))

        performance["Risk"] = np.append(performance["Risk"], annualized_risk)

        sharpe_ratio = cumulated_return / annualized_risk

        performance["Sharpe-Ratio"] = np.append(performance["Sharpe-Ratio"], sharpe_ratio)
    performance_strategy = pd.DataFrame.from_dict(performance).transpose()
    print(performance_strategy)


def coin_portfolio(data):
    #Evaluation of the news Stratygy

    mean_return_per_day = data[data['Strategy1'] != 0].groupby([
        'start'])['Strategy1'].mean().reset_index()['Strategy1']

    # since there may be days where the algorithm decides not to trade at all,
    # the zero returns of the days without trading are added to the means per day.
    # this way the mean return & volatility are comparable to the market

    zeros = pd.DataFrame(np.repeat(0, (data['start'].nunique() - len(mean_return_per_day))))
    means = np.append(mean_return_per_day, zeros)

    returns = np.nanmean(means)
    risk = np.nanstd(means, ddof=0)
    sharpe_ratio = (returns / risk)

    print(f"The mean return per day is: {returns}, the risk is: {risk} and the sharpe ratio is: {sharpe_ratio}")

def coins_portfolio_long(data):
    returns_market = data.groupby(['start'])['ret'].mean().mean()
    risk_market = data.groupby(['start'])['ret'].mean().std(ddof=0)
    sharpe_ratio_market = (returns_market / risk_market)

    print(
        f"The mean return per day is: {returns_market}, the risk is: {risk_market} and the sharpe ratio is: {sharpe_ratio_market}")

def crypto_index(ticker):
    #^ CIX10
    # for comparison also the results of the CIX100 index are calculated
    CIX100 = yf.download(ticker, start="2019-10-01", end="2020-01-01")
    CIX100 = CIX100.reset_index()
    CIX100['ret'] = (((CIX100['Adj Close'] / CIX100['Adj Close'].shift(1)) - 1) * 100)
    CIX100_return = CIX100['ret'].mean()
    CIX100_volatility = CIX100['ret'].std(ddof=0)
    CIX100_sharpe_ratio = CIX100_return / CIX100_volatility
    print(CIX100_return, CIX100_volatility, CIX100_sharpe_ratio)
    return CIX100

def comparison(data, comparable):
    # calculate the excess return in comparison to sp500 per day
    data.rename(columns = {'start':'Date'}, inplace=True)
    strategy1_return = data[data["Strategy1"] != 0].groupby('Date')[
        'Strategy1'].mean()
    strategy1_return = pd.DataFrame(strategy1_return).reset_index()
    excess = strategy1_return.merge(comparable[["Date", "ret"]], on='Date', how="right")
    excess = excess.fillna(0)
    excess["excess_return"] = excess["Strategy1"] - excess["ret"]
    dict1 = {'Cum_Ret_Strategy': data[data["Strategy1"] != 0].groupby(
        ['Date'])["Strategy1"].mean().cumsum()}
    dict2 = {'Date': comparable.Date, 'Cum_Ret_SP500': comparable["ret"].cumsum()}
    dict3 = {'Cum_Ret_SMA': data.groupby(['Date'])['StrategySMA'].mean().cumsum()}
    dict4 = {'Date': excess.Date, 'Cum_Ret_Excess': excess["excess_return"].cumsum()}
    dict5 = {'Cum_Ret_BuyAll': data.groupby(['Date'])['ret'].mean().cumsum()}

    time = lambda x: x + datetime.timedelta(hours = -8)
    df1 = pd.DataFrame.from_dict(dict1).reset_index()
    df1.info()
    df1.Date = df1.Date.apply(time)
    df1.info()
    df2 = pd.DataFrame.from_dict(dict2)
    df3 = pd.DataFrame.from_dict(dict3).reset_index()
    df3.Date = df3.Date.apply(time)
    df4 = pd.DataFrame.from_dict(dict4)
    df5 = pd.DataFrame.from_dict(dict5).reset_index()
    df5.Date = df5.Date.apply(time)

    plot_data = df2.merge(df1, on='Date', how='left')
    plot_data =  plot_data.merge(df3, on='Date').merge(df4, on='Date', how='left').merge(
        df5, on="Date", how='left')
    plot_data["Excess_News_to_Market"] = plot_data["Cum_Ret_Strategy"] - plot_data["Cum_Ret_BuyAll"]
    # lot_data.loc[:, plot_data.columns != "Date"].divide(100).add(1)
    print(plot_data.head())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_data.Date, y=plot_data.Cum_Ret_SP500, name="SP500",
                             line=dict(color='black', width=2)))
    fig.add_trace(go.Scatter(x=plot_data.Date, y=plot_data.Cum_Ret_Strategy, name="Algorithm Strategy",
                             mode='lines', line=dict(color='blue', width=2)))
    fig.add_trace(
        go.Scatter(x=plot_data.Date, y=plot_data.Cum_Ret_Excess, mode='lines', name="Excess Return", line=dict(
            color='lightblue', width=2)))
    fig.add_trace(go.Scatter(x=plot_data.Date, y=plot_data.Cum_Ret_SMA, name='SMA-Strategy', mode='lines',
                             line=dict(color='pink', width=2)))
    fig.add_trace(go.Scatter(x=plot_data.Date, y=plot_data.Cum_Ret_BuyAll, name='Buy-all', mode='lines',
                             line=dict(color='grey', width=2)))
    fig.update_layout(title='Algorithm vs Market',
                      xaxis_title='Date',
                      yaxis_title='Cumulated Returns in %',
                      paper_bgcolor='rgba(100,100,100)',
                      plot_bgcolor='rgb(240,248,255)')

    fig.show()
    plt.show()








long_coins()
sentiment_strategy()
coin_portfolio(backtesting)
coins_portfolio_long(backtesting)
bitcoin = crypto_index('BTC-USD')
comparison(backtesting, bitcoin)
