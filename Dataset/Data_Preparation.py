import pandas as pd
import numpy as np

pd.options.display.width= None
pd.options.display.max_columns= None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

'''
The different downloaded Datasets are merged in this file and the final dataset which is used for the 
regression analysis in the next steps is formed
'''


sentiment_reddit_df = pd.read_csv('/Users/torbenleowald/Documents/Python Finance/python_trading_bot/Dataset/reddit.csv')
google_trend = pd.read_csv('/Users/torbenleowald/Documents/Python Finance/python_trading_bot/Dataset/google_trends.csv')
# data cleaning and preparation for google trend data
google_trend.fillna(method='ffill', inplace=True)
google_trend.fillna(0, inplace=True)
google_trend.set_index('date', inplace=True)
# google_trend.drop(google_trend.columns[0], inplace = True)
google_trend = google_trend.stack().reset_index()
google_trend.rename(columns={'level_1': 'coin'}, inplace=True)
# calculate lagged returns as additional independent variables used in the regression
def lags(data, number_lags):
    for lag in range(1, number_lags+1):
        col = 'lag_{}'.format(lag)
        data[lag] = data['ret'].shift(lag)

# merge all dataframes together to one large dataset
merged = sentiment_reddit_df.merge(google_trend, left_on=['Date', 'coin'], right_on=['date', 'coin'], how='left')
merged.rename(columns={0: 'google_trend'}, inplace=True)
lags(merged, 4)
merged['momentum'] = merged['ret'].rolling(5).mean().shift(1)
merged.dropna(inplace = True)
merged['google_trend_change'] = merged['google_trend'].pct_change()
merged['google_trend_change'].replace([np.inf, -np.inf], inplace = True)
merged['google_trend_change'].fillna(method ='bfill', inplace = True)
merged['score_change'] = merged['score'].pct_change()
merged['score_change'].replace([np.inf, -np.inf], inplace = True)
merged['score_change'].fillna(method ='bfill', inplace = True)
merged['google_trend_log'] = np.log(merged['google_trend'])
merged['ret_average'] = merged['ret']*merged['average']
merged['ret_score'] = merged['ret']*merged['score']
merged['ret_google_trend'] = merged['ret']*merged['google_trend']
merged['vola_average'] = merged['vola']*merged['average']
merged['vola_score'] = merged['vola']*merged['score']
merged['vola_google_trend'] = merged['vola']*merged['google_trend']
merged['ret_squared'] = merged['ret']*merged['ret']
merged['ret_log'] = np.log(merged['ret'])
merged['ret_log'].replace([np.inf, -np.inf], inplace = True)
merged['ret_log'].fillna(method = 'bfill', inplace = True)
merged['ret_log'].fillna(method = 'ffill', inplace = True)
merged.to_csv('merged.csv')
print(merged[['ret', 'score',  'google_trend', 'volumeUSD', 'vola', 'ret_google_trend', 'vola_score']].describe())
