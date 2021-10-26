from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn import preprocessing as prep
from Dataset.Data_download import urls_to_df_from_url

def svm(data):
    split = int(len(data) * 0.8)
    train = data.iloc[:split].copy()
    test = data.iloc[split:].copy()
    train_x = train.filter(['return']).to_numpy()
    test_x = test.filter(['return']).to_numpy()
    train_y = train[['average', 'score', 'ret']].to_numpy()
    test_y = test[['average', 'score', 'ret']].to_numpy()
    train_y = prep.normalize(train_y, norm='l2')
    test_y = prep.normalize(test_y, norm='l2')
    test_x = prep.normalize(test_x, norm='l2')
    train_x = prep.normalize(train_x, norm='l2')
    import warnings
    warnings.filterwarnings('ignore')
    for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
        model = SVC(kernel=kernel)
        model.fit(train_y, train_x)
        print('{:>8s} | {:8.3f}'.format(kernel, accuracy_score(test_x, model.predict(test_y))))


urls = ['ada', 'algo', 'atom','bat', 'bch', 'bnb', 'btc', 'cvc', 'dai', 'dash', 'dnt', 'doge', 'eos', 'gnt', 'knc',
        'link', 'loom', 'ltc', 'mana', 'mkr', 'neo', 'rep', 'trx', 'xem', 'xlm', 'xrp', 'xtz', 'zec', 'zrx']
sentiment_df = urls_to_df_from_url(urls, 'reddit').dropna()
svm(sentiment_df)
