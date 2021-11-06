import yfinance as yf
import datetime as dt


# helper file to download conveniently from yahoo finance



def download_ticker(ticker):
    end = dt.datetime(2019, 10,1)
    df = yf.download(ticker, end = end, show_errors=True)
    df = df.reset_index()
    df['ret'] = (((df['Adj Close'] / df['Adj Close'].shift(1)) - 1) * 100)
    ticker_return = df['ret'].mean()
    ticker_volatility = df['ret'].std(ddof=0)
    ticker_sharpe_ratio = ticker_return / ticker_volatility
    print(ticker_return, ticker_volatility, ticker_sharpe_ratio)
    return df
