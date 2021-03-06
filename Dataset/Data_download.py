import numpy as np
import pandas as pd
import datetime


pd.options.display.width= None
pd.options.display.max_columns= None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)
'''
In this python file the Sentiment Dataset is downloaded and the different columns prepared
'''

def git_download(path, name):
    # change millisecond timestamp to dateobject
    f = lambda ms: datetime.datetime.fromtimestamp(int(ms) / 1000.0)
    df = pd.read_csv(path, index_col=1, date_parser=f)
    # check number of nan values in the sentiment score column, if the column has more than 100 nan values we do not
    # include the coin since the amount of data is too little for the later regression steps
    if df.score.isnull().sum() > 100:
        return
    # handle nan data
    else:
        # nan values in the sentiment column are filled with the values of the day before since we assume the sentiment
        # has most likely stayed constant or its true value is at least similar to the one of the day before
        df.score.fillna(method = 'ffill', inplace=True)
    df['coin'] = name
    # calculate return per day
    df['ret'] = df.close.pct_change()
    df.ret.fillna(0, inplace=True)
    #df.ret = df.ret.clip(lower=df.ret.quantile(0.005), upper=df.ret.quantile(0.995))
    df['ret_1'] = df.ret.shift(-1)
    df.ret_1.fillna(0, inplace=True)
    # create decision variable 1 for positive return and 0 for negative returns of the next day
    df['return'] = np.sign(df['ret_1'])
    df['vola'] = (df.ret.rolling(window=30).std())*(30)**1/2
    # same reasoning for ffill as for the sentiment column
    df.average.fillna(method='ffill', inplace=True)
    df.positive.fillna(method='ffill', inplace=True)
    df.negative.fillna(method='ffill', inplace=True)

    df.reset_index(inplace = True)
    df.rename(columns={'start': 'Date'}, inplace=True)
    # change the datetime object from 8 in the morning to midnight to enable merging at a later stage
    time = lambda x: x + datetime.timedelta(hours=-8)
    df.Date = pd.to_datetime(df.Date)
    df.Date = df.Date.apply(time)
    return df


def urls_to_df_from_url(urls, platform):
    content = []
    for url in urls:
        if platform == 'reddit':
            content.append(git_download('https://github.com/Qokka/crypto-sentiment-data/blob/ee96212e68e1796b8bb14cdae7e7bacf6d78eed0/2019-daily-sentiment-price/' + url + '_reddit.csv?raw=true', url))
        else:
            content.append(git_download(
                'https://github.com/Qokka/crypto-sentiment-data/blob/ee96212e68e1796b8bb14cdae7e7bacf6d78eed0/2019-daily-sentiment-price/' + url + '_telegram.csv?raw=true', url) )
    df = pd.concat(content, sort=False)
    return df

urls = ['ada', 'algo', 'atom','bat', 'bch', 'bnb', 'btc', 'cvc', 'dash', 'dnt', 'doge', 'eos', 'gnt', 'knc',
        'link', 'loom', 'ltc', 'mana', 'mkr', 'neo', 'rep', 'trx', 'xem', 'xlm', 'xrp', 'xtz', 'zec', 'zrx']
urls_telegram = ['ada', 'algo', 'atom', 'dnt', 'eos','knc', 'loom', 'mana', 'trx', 'xem', 'xrp', 'xtz']
df_reddit = urls_to_df_from_url(urls, 'reddit')
df_telegram = urls_to_df_from_url(urls_telegram, 'telegram')

df_reddit.to_csv('reddit.csv')
df_telegram.to_csv('telegram.csv')