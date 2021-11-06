import datetime

import numpy as np
import yfinance as yf
from Strategy import predictions as backtesting
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt


''' This class implements the evaluation of the different stragies and plots the result as cummulative returns when it 
is executed
'''

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

        ### calculating the sum of the returns
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
        cumulated_return = backtesting.loc[(backtesting['coin'] == coin), 'Strategy1'].cumsum(
        ).apply(np.exp).iloc[-1] - 1
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
    #Evaluation of the Sentiment Stratygy

    mean_return_per_day = data[data['Strategy1'] != 0].groupby([
        'Date'])['Strategy1'].mean().reset_index()['Strategy1']

    # since there may be days where the algorithm decides not to trade at all,
    # the zero returns of the days without trading are added to the means per day.
    # this way the mean return & volatility are comparable to the market

    zeros = pd.DataFrame(np.repeat(0, (data['Date'].nunique() - len(mean_return_per_day))))
    means = np.append(mean_return_per_day, zeros)

    returns = np.nanmean(means)
    risk = np.nanstd(means, ddof=0)
    sharpe_ratio = (returns / risk)

    print(f"The mean return per day is: {returns}, the risk is: {risk} and the sharpe ratio is: {sharpe_ratio}")

def coins_portfolio_long(data):
    returns_market = data.groupby(['Date'])['ret'].mean().mean()
    risk_market = data.groupby(['Date'])['ret'].mean().std(ddof=0)
    sharpe_ratio_market = (returns_market / risk_market)

    print(
        f"The mean return for holding all coins per day is: {returns_market}, the risk is: {risk_market} and the sharpe ratio is: {sharpe_ratio_market}")

def crypto_index(ticker):
    # for comparison also the results of the CIX100 index are calculated
    comparable = yf.download(ticker, start="2019-10-01", end="2020-01-01")
    comparable = comparable.reset_index()
    comparable['ret'] = (((comparable['Adj Close'] / comparable['Adj Close'].shift(1)) - 1) * 100)
    comparable_return = comparable['ret'].mean()
    comparable_volatility = comparable['ret'].std(ddof=0)
    comparable_sharpe_ratio = comparable_return / comparable_volatility
    print(comparable_return, comparable_volatility, comparable_sharpe_ratio)
    return comparable

def comparison(data, comparable):
    # calculate the excess return in comparison to sp500 per day
    strategy1_return = data[data["Strategy1"] != 0].groupby('Date')[
        'Strategy1'].mean()
    strategy1_return = pd.DataFrame(strategy1_return).reset_index()
    strategy1_return.Date = pd.to_datetime(strategy1_return.Date)
    excess = strategy1_return.merge(comparable[["Date", "ret"]], on='Date', how="right")
    excess = excess.fillna(0)
    excess["excess_return"] = excess["Strategy1"] - excess["ret"]
    dict1 = {'Cum_Ret_Strategy': data[data["Strategy1"] != 0].groupby(
        ['Date'])["Strategy1"].mean()}
    dict2 = {'Date': comparable.Date, 'Cum_Ret_SP500': comparable["ret"]}
    dict3 = {'Cum_Ret_SMA': data.groupby(['Date'])['StrategySMA'].mean()}
    dict4 = {'Date': excess.Date, 'Cum_Ret_Excess': excess["excess_return"]}
    dict5 = {'Cum_Ret_BuyAll': data.groupby(['Date'])['ret'].mean()}

    df1 = pd.DataFrame.from_dict(dict1).reset_index()
    df1.Date = pd.to_datetime(df1.Date)
    df2 = pd.DataFrame.from_dict(dict2)
    df3 = pd.DataFrame.from_dict(dict3).reset_index()
    df3.Date = pd.to_datetime(df3.Date)
    df4 = pd.DataFrame.from_dict(dict4)
    df5 = pd.DataFrame.from_dict(dict5).reset_index()
    df5.Date = pd.to_datetime(df5.Date)

    plot_data = df2.merge(df1, on='Date', how='left')
    plot_data =  plot_data.merge(df3, on='Date').merge(df4, on='Date', how='left').merge(
        df5, on="Date", how='left')
    plot_data["Excess_News_to_Market"] = plot_data["Cum_Ret_Strategy"] - plot_data["Cum_Ret_BuyAll"] + 1
    plot_data['Cum_Ret_SP500'] = plot_data['Cum_Ret_SP500']/100
    plot_data.set_index('Date', inplace = True)
    plot_data[['Cum_Ret_SP500', 'Cum_Ret_Strategy', 'Cum_Ret_SMA', 'Cum_Ret_BuyAll']].dropna().cumsum(
    ).apply(np.exp).plot(figsize=(10, 6))
    plot_data['Cum_Ret_SP500'] = plot_data['Cum_Ret_SP500'].dropna().cumsum(
    ).apply(np.exp)
    plot_data['Cum_Ret_Strategy'] = plot_data['Cum_Ret_Strategy'].dropna().cumsum(
    ).apply(np.exp)
    plot_data['Cum_Ret_SMA'] = plot_data['Cum_Ret_SMA'].dropna().cumsum(
    ).apply(np.exp)
    plot_data['Cum_Ret_BuyAll'] = plot_data['Cum_Ret_BuyAll'].dropna().cumsum(
    ).apply(np.exp)
    plot_data.reset_index(inplace = True)

    # lot_data.loc[:, plot_data.columns != "Date"].divide(100).add(1)
    print(plot_data.head())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_data.Date, y=plot_data.Cum_Ret_SP500, name="Bitcoin",
                             line=dict(color='black', width=2)))
    fig.add_trace(go.Scatter(x=plot_data.Date, y=plot_data.Cum_Ret_Strategy, name="Algorithm Strategy",
                             mode='lines', line=dict(color='blue', width=2)))
    fig.add_trace(
        go.Scatter(x=plot_data.Date, y=plot_data.Excess_News_to_Market, mode='lines', name="Excess Return", line=dict(
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

def facts(data):
    # calculate the excess return in comparison to sp500 per day
    data.rename(columns={'start': 'Date'}, inplace=True)
    strategy1_return = data[data["Strategy1"] != 0].groupby('Date')[
        'Strategy1'].mean()
    strategy1_return = pd.DataFrame(strategy1_return).reset_index()
    strategy1_return.Date = pd.to_datetime(strategy1_return.Date)
    excess = strategy1_return.merge(bitcoin[["Date", "ret"]], on='Date', how="right")
    excess = excess.fillna(0)
    excess["excess_return"] = excess["Strategy1"] - excess["ret"]
    ### cumulated return is the sum of the mean returns per day
    cumulated_returns = data[data["Strategy1"] != 0].groupby(['Date'])[
        "Strategy1"].mean().sum()

    ### excess return is the sum of the daily excess returns in comparison to SP500
    excess_return = excess["excess_return"].sum()

    ### the number of trades is the number of times the position was not equal to 0
    number_of_trades = data[data['Position_Strategy1'] != 0].count()[0]

    ### number of wins/losses is the nr. of times the strategy had pos./neg. returns for a single trade
    number_of_wins = data[data['Strategy1'] > 0].count()[0]
    number_of_losses = data[data['Strategy1'] < 0].count()[0]

    ### days with pos. / neg. returns are the days where the mean return was bigger/smaller 0
    days_positive = sum(data[data["Strategy1"] != 0].groupby(['Date'])[
                            "Strategy1"].mean() > 0)
    days_negative = sum(data[data["Strategy1"] != 0].groupby(['Date'])[
                            "Strategy1"].mean() < 0)

    facts = {'Cumulated Returns': cumulated_returns, 'Number of trades': number_of_trades,
             "Excess Returns": excess_return,
             'Number of wins': number_of_wins, 'Number of losses': number_of_losses,
             'Number of days with pos. Returns': days_positive,
             'Number of days with neg. Returns': days_negative}
    facts = pd.DataFrame.from_dict(facts, orient='index')
    facts = facts.reset_index()
    facts = facts.rename(columns={'index': 'Statistic', 0: 'Value'})
    print(facts)

def more_facts(data):
    ### this dataframe shows how often each coin was traded by the strategy
    print(data[data['Position_Strategy1'] != 0][
        'coin'].value_counts())







long_coins()
sentiment_strategy()
coin_portfolio(backtesting)
coins_portfolio_long(backtesting)
bitcoin = crypto_index('BTC-USD')
comparison(backtesting, bitcoin)
facts(backtesting)
more_facts(backtesting)
