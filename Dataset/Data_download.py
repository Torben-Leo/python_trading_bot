import requests
import json
import os
from io import StringIO
import pandas as pd
import glob
import datetime
from pathlib import Path
#?raw=true

#todo: Data cleaning, delete cc's without values in the sentiment columns, build first regression models,
# is sentimenet score allone enough as predictor variable


# datetime.datetime.fromtimestamp(ms/1000.0)

def git_download(path):
    # change millisecond timestamp to dateobject
    f = lambda ms: datetime.datetime.fromtimestamp(int(ms) / 1000.0)
    df = pd.read_csv(path, index_col=1, date_parser=f)
    #check number of nan values in the sentiment score column
    if df.score.isnull().sum() > 100:
        return
    # handle nan data
    else:
        df.fillna(method = 'ffill')
    return df


def urls_to_df_from_url(urls):
    content = []
    for url in urls:
        content.append(git_download('https://github.com/Qokka/crypto-sentiment-data/blob/ee96212e68e1796b8bb14cdae7e7bacf6d78eed0/2019-daily-sentiment-price/' + url + '_reddit.csv?raw=true'))
    df = pd.concat(content, sort=False)
    return df

urls = ['ada', 'algo', 'atom','bat', 'bch', 'bnb', 'btc', 'cvc', 'dai', 'dash', 'dnt', 'doge', 'eos', 'gnt', 'knc',
        'link', 'loom', 'ltc', 'mana', 'mkr', 'neo', 'rep', 'trx', 'xem', 'xlm', 'xrp', 'xtz', 'zec', 'zrx']
df = urls_to_df_from_url(urls)
print(df.head())
print(df.info())
print(df.describe())

