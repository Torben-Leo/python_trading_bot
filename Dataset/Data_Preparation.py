import pandas as pd

sentiment_reddit_df = pd.read_csv('/Users/torbenleowald/Documents/Python Finance/python_trading_bot/Dataset/reddit.csv')
google_trend = pd.read_csv('/Users/torbenleowald/Documents/Python Finance/python_trading_bot/Dataset/google_trends.csv')
google_trend.fillna(method='ffill', inplace=True)
google_trend.fillna(0, inplace=True)
google_trend.set_index('date', inplace=True)
# google_trend.drop(google_trend.columns[0], inplace = True)
google_trend = google_trend.stack().reset_index()
google_trend.rename(columns={'level_1': 'coin'}, inplace=True)
def lags(data, number_lags):
    for lag in range(1, number_lags+1):
        col = 'lag_{}'.format(lag)
        data[lag] = data['ret'].shift(lag)


merged = sentiment_reddit_df.merge(google_trend, left_on=['Date', 'coin'], right_on=['date', 'coin'], how='left')
merged.rename(columns={0: 'google_trend'}, inplace=True)
lags(merged, 4)
merged['momentum'] = merged['ret'].rolling(5).mean().shift(1)
merged.dropna(inplace = True)

merged.to_csv('merged.csv')
