import pandas as pd

sentiment_reddit_df = pd.read_csv('/Users/torbenleowald/Documents/Python Finance/python_trading_bot/Dataset/reddit.csv')
google_trend = pd.read_csv('/Users/torbenleowald/Documents/Python Finance/python_trading_bot/Dataset/google_trends.csv')
google_trend.fillna(method='ffill', inplace=True)
google_trend.fillna(0, inplace=True)
google_trend.set_index('date', inplace=True)
# google_trend.drop(google_trend.columns[0], inplace = True)
google_trend = google_trend.stack().reset_index()
google_trend.rename(columns={'level_1': 'coin'}, inplace=True)

merged = sentiment_reddit_df.merge(google_trend, left_on=['Date', 'coin'], right_on=['date', 'coin'], how='left')
merged.rename(columns={0: 'google_trend'}, inplace=True)
merged.to_csv('merged.csv')
