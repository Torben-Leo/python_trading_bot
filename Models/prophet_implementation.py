from prophet import Prophet
from prophet.plot import plot_plotly
from prophet.plot import add_changepoints_to_plot
import pandas as pd
from Dataset.timeseries_data import downlaod_tickers
import matplotlib.pyplot as plt


def prediction(data):
    # reset index for Date column
    df = data.reset_index()
    # create a dataframe
    train_dataset = pd.DataFrame()

    train_dataset['ds'] = df['Date']
    train_dataset['y'] = df['Return']
    train_dataset.head(2)
    # create a Prophet instance with default values to fit the dataset
    prophet_basic = Prophet()
    prophet_basic.fit(train_dataset)
    # create a dataframe with ds (i.e., datetime stamp) that has the time series of dates we need for prediction
    # periods specify the number of days to extend into the future
    future = prophet_basic.make_future_dataframe(periods=365)
    future.tail()
    # forecast BTC prices
    forecast = prophet_basic.predict(future)
    # plot predicted BTC prices
    fig1 = prophet_basic.plot(forecast)
    # plot the trend and seasonality
    fig1 = prophet_basic.plot_components(forecast)
    # identify changepoints (i.e., datetime points when the time series exprience abrupt changes)

    fig = prophet_basic.plot(forecast)
    a = add_changepoints_to_plot(fig.gca(), prophet_basic, forecast)
    print(prophet_basic.changepoints)
    # adjust trend sensitivity with "changepoint_prior_scale" parameter
    # default value is 0.05. Lower value, less flexible trend, and vice versa

    pro_change = Prophet(changepoint_prior_scale=0.15)
    forecast = pro_change.fit(train_dataset).predict(future)
    fig = pro_change.plot(forecast)
    a = add_changepoints_to_plot(fig.gca(), prophet_basic, forecast)
    plt.show()


prediction(downlaod_tickers(['BTC-USD']))