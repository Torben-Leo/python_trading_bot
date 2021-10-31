import pandas as pd
from pytrends.request import TrendReq
import plotly.express as px
import time
from pytrendsdaily import getDailyData

startTime = time.time()

pytrends = TrendReq(hl='en-US', tz=360, retries=10, backoff_factor=0.5)

# build payload
def download_googletrends(names, tickers):
    df = pd.DataFrame()
    df['date'] = pd.date_range('2019-01-01', periods=366, freq='D')
    i = 0
    for name in names:
        try:
            pytrends.build_payload([name], cat=0, timeframe='2019-01-01 2020-01-01', geo='', gprop='')
            data = pytrends.interest_over_time()
            #data = getDailyData(name, 2019)
            data = data.reset_index()
            data.rename(columns = {name:tickers[i]}, inplace = True)
            df = df.merge(data[['date', tickers[i]]], on = 'date' , how = 'left')
        except Exception as e:
            print(name, "couldn't be done, because " + str(e))
        i +=1
    return df

names = ['Cardano', 'Algorand', 'Cosmos', 'Basic Attention Token', 'Bitcoin Cash', 'Binance', 'Bitcoin', 'Civic',
         'Dai',  'Dash', 'district0x', 'Doge', 'Eos', 'GreenTrust', 'Kyber', 'Chainlink', 'Loom', 'Litecoin',
         'Decentraland', 'Maker', 'Neo', 'Augur', 'Tron', 'Nem', 'Stellar', 'Ripple', 'Tezos', 'Zcash', '0x']
ticker = ['ada', 'algo', 'atom','bat', 'bch', 'bnb', 'btc', 'cvc', 'dai', 'dash', 'dnt', 'doge', 'eos', 'gnt', 'knc',
        'link', 'loom', 'ltc', 'mana', 'mkr', 'neo', 'rep', 'trx', 'xem', 'xlm', 'xrp', 'xtz', 'zec', 'zrx']
df = download_googletrends(names, ticker)
df_csv = df.to_csv('google_trends.csv')