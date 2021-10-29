import pandas as pd
from pytrends.request import TrendReq
import plotly.express as px
import time
startTime = time.time()

pytrends = TrendReq(hl='en-US', tz=360, retries=10, backoff_factor=0.5)

# build payload
def download_googletrends(names):
    df = pd.DataFrame()
    for name in names:
        try:
            pytrends.build_payload(name, cat=0, timeframe='2019-01-01 2020-01-01', geo='', gprop='')
            data = pytrends.interest_over_time()
            data = data.reset_index()
            df.concat(data)
        except:
            print(name, "couldn't be done")
    return df

names = ['Cardano', 'Algorand', 'Cosmos', 'Basic Attention Token', 'Bitcoin Cash', 'Binance', 'Bitcoin', 'Civic',
         'Dai',  'Dash', 'district0x', 'Doge', 'Eos', 'GreenTrust', 'Kyber', 'Chainlink', 'Loom', 'Litecoin',
         'Decentraland', 'Maker', 'Neo', 'Augur', 'Tron', 'Nem', 'Stellar', 'Ripple', 'Tezos', 'Zcash', '0x']
df = download_googletrends(names)
df_csv = df.to_csv('google_trends.csv')


