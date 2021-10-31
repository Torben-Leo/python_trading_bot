import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn import preprocessing as prep


def svm(data, variables):
    train = data[data.Date < "2019-10-01"]
    test = data[data.Date >= "2019-10-01"]
    train_x = train.filter(['return']).to_numpy()
    test_x = test.filter(['return']).to_numpy()
    train_y = train[variables].to_numpy()
    test_y = test[variables].to_numpy()
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


variables = ['average', 'ret', 'google_trend', 'score', '1', '2', '3', '4', 'vola']
merged = pd.read_csv('/Users/torbenleowald/Documents/Python Finance/python_trading_bot/Dataset/merged.csv')
svm(merged)
