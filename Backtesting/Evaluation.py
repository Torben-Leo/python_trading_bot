import numpy as np
from Strategy import predictions as backtesting
import pandas as pd


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



long_coins()
sentiment_strategy()
coin_portfolio(backtesting)
coins_portfolio_long(backtesting)
