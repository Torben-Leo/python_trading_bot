import tensorflow as tf
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.optimizers import Adam
import numpy as np
import random
import pandas as pd

def set_seeds(seed = 100):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def tensor_flow(df, variables):
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001)
    set_seeds()
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(64, activation = 'relu', input_shape=(len(variables,))))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer= optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    model_data = df[df.Date < "2019-10-01"]
    backtesting_data = df[df.Date >= "2019-10-01"]
    mu, std = model_data.mean(), model_data.std()
    model_data_ = (model_data-mu)/std
    backtesting_data_ = (backtesting_data-mu)/std
    model.fit(model_data[variables], model_data['return'], epochs=50,
              verbose = False, validation_split=0.2, shuffle=False)
    print(model.evaluate())


variables = ['average', 'ret', 'google_trend', 'score', 'momentum', '1', '2', '3', '4', 'vola']
merged = pd.read_csv('/Users/torbenleowald/Documents/Python Finance/python_trading_bot/Dataset/merged.csv')
tensor_flow(merged, variables)