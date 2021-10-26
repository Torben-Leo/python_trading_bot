from sklearn. tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from Dataset.Data_download import urls_to_df_from_url





def random_forest(df):
    # check for distribution of target variable:
    print(df.ret_1.value_counts() / len(df.ret_1))
    # make train- test split based on date, not random
    model_data = df[df.start < "2019-10-01"]
    backtesting_data = df[df.start >= "2019-10-01"]
    variables = ['average', 'score', 'ret']
    # Split the data into explanatory and dependable variables for test and train datasets
    X = model_data[['average', 'score', 'ret']]
    y = model_data['return']

    X_test = backtesting_data[['average', 'score', 'ret']]
    y_test = backtesting_data['return']
    '''
    n_estimators = [5, 10, 50, 100, 250]
    max_depth = [2, 4, 8, 10, 12, 14, 16, 18, 20, 32, None]
    min_samples_split = [2, 5, 7, 10, 13, 20]

    max_score = 0
    params = []

    for n_e in n_estimators:
        for max_d in max_depth:
            for min_s in min_samples_split:
                model = RandomForestClassifier(max_depth=max_d, n_estimators=n_e, min_samples_split=min_s,
                                               random_state=4321)
                model = model.fit(X, y)
                score = model.score(X_test, y_test)
                if score > max_score:
                    max_score = score
                    params = model.get_params()
                    print(score)

    print(params)
    '''
    model = RandomForestClassifier(max_depth=4, n_estimators=50, min_samples_split=7, random_state=4321)
    model = model.fit(X, y)
    prediction = model.predict(X_test)
    print(model.score(X_test, y_test))
    features = model.feature_importances_
    plt.barh(variables, features)
    plt.show()



urls = ['ada', 'algo', 'atom','bat', 'bch', 'bnb', 'btc', 'cvc', 'dai', 'dash', 'dnt', 'doge', 'eos', 'gnt', 'knc',
        'link', 'loom', 'ltc', 'mana', 'mkr', 'neo', 'rep', 'trx', 'xem', 'xlm', 'xrp', 'xtz', 'zec', 'zrx']
sentiment_df = urls_to_df_from_url(urls, 'reddit').dropna()
random_forest(sentiment_df)