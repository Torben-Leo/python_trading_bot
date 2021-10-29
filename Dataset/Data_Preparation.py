import pandas as pd


reddit = pd.read_csv('reddit.csv')
telegram = pd.read_csv('telegram.csv')
#google_trend = pd.read_csv('google_trends.csv')

print(reddit.head())
print(telegram.head())
#print(google_trend.head())

data = reddit.merge(telegram[['coin', 'score']], how = 'left', on = 'coin')
print(data.head())
