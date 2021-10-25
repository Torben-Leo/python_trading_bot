from sklearn. tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from Dataset.Data_download import urls_to_df_from_url





def random_forest(df):
    # check for distribution of target variable:
    print(df.ret_1.value_counts() / len(df.ret_1))
    # make train- test split based on date, not random
    model_data = df[df.index < "2019-10-01"]
    backtesting_data = df[df.index >= "2019-10-01"]
    variables = ['score', 'volumeUSD', 'vola']
    # Split the data into explanatory and dependable variables for test and train datasets
    X = model_data[['score', 'volumeUSD', 'vola']]
    y = model_data['ret_1']

    X_test = backtesting_data[['score', 'volumeUSD', 'vola']]
    y_test = backtesting_data['ret_1']
    model = RandomForestClassifier(max_depth=14, n_estimators=250, min_samples_split=2, random_state=4321)
    model = model.fit(X, y)
    prediction = model.predict(X_test)
    print(model.score(X_test, y_test))
    features = model.feature_importances_
    plt.barh(variables, features)


urls = ['ada', 'algo', 'atom','bat', 'bch', 'bnb', 'btc', 'cvc', 'dai', 'dash', 'dnt', 'doge', 'eos', 'gnt', 'knc',
        'link', 'loom', 'ltc', 'mana', 'mkr', 'neo', 'rep', 'trx', 'xem', 'xlm', 'xrp', 'xtz', 'zec', 'zrx']
sentiment_df = urls_to_df_from_url(urls).dropna()
random_forest(sentiment_df)