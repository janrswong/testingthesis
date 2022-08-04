# TODO: FIX LSTM model summary show
from statsmodels.tsa.arima_model import ARIMAResults
import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import joblib
import math
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras import layers
from keras import wrappers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# page expands to full width
st.set_page_config(page_title="LSTM vs ARIMA", layout='wide')

# PAGE LAYOUT
# heading
st.title("Make a Model")

# ARIMA PARAMETERS
pValue = 4
dValue = 1
qValue = 0

# show raw data
st.header("Raw Data")
# sidebar
# Sidebar - Specify parameter settings
with st.sidebar.header('Set Data Split'):
  # PARAMETERS min,max,default,skip
    trainData = st.sidebar.slider(
        'Data split ratio (% for Training Set)', 10, 90, 80, 5)
    # ARIMA PARAMETERS
    pValue = st.sidebar.number_input('P-value:', 0, 100, pValue)
    st.sidebar.write('The current p-Value is ', pValue)
    dValue = st.sidebar.number_input('D-value:', 0, 100, dValue)
    st.sidebar.write('The current d-Value is ', dValue)
    qValue = st.sidebar.number_input('Q-value:', 0, 100, qValue)
    st.sidebar.write('The current q-Value is ', qValue)


# select time interval
interv = st.select_slider('Select Time Series Data Interval for Prediction', options=[
                          'Weekly', 'Monthly', 'Quarterly', 'Yearly'])


def getInterval(argument):
    switcher = {
        "W": "1wk",
        "M": "1mo",
        "Q": "3mo",
        "Y": "1d"
    }
    return switcher.get(argument, "1d")


df = yf.download('BZ=F', interval=getInterval(interv[0]))
st.table(df.head())
# download full data


@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


csv = convert_df(df)

st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='Brent Oil Prices.csv',
    mime='text/csv',
)
# download full data


# graph visualization
st.header("Visualizations")

# LSTM


@st.cache
def df_to_X_y(df, window_size=5):
    df_as_np = df.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np)-window_size):
        row = [[a] for a in df_as_np[i:i+window_size]]
        X.append(row)
        label = df_as_np[i+window_size]
        y.append(label)
    return np.array(X), np.array(y)


@st.cache
def mse_eval(test, predictions):
    return mean_squared_error(test, predictions)


@st.cache
def mape_eval(test, predictions):
    return mean_absolute_percentage_error(test, predictions)


@st.cache(allow_output_mutation=True)
def evaluate_lstm_model(split):
    global lstmModel
    WINDOW_SIZE = 3
    X1, y1 = df_to_X_y(df['Close'], WINDOW_SIZE)

    # preprocessing
    date_train, date_test = df.index[:int(
        df.shape[0]*split)], df.index[int(df.shape[0]*split)+WINDOW_SIZE:]
    X_train1, y_train1 = X1[:int(df.shape[0]*split)
                            ], y1[:int(df.shape[0]*split)]
    X_test1, y_test1 = X1[int(df.shape[0]*split):], y1[int(df.shape[0]*split):]
    # X_train1.shape, y_train1.shape, X_test1.shape, y_test1.shape

    # lstm model
    model = Sequential([layers.Input((3, 1)), layers.LSTM(64), layers.Dense(
        32, activation='relu'), layers.Dense(32, activation='relu'), layers.Dense(1)])
    cp1 = ModelCheckpoint('model1/', save_best_only=True)
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001),
                  metrics=['mean_absolute_percentage_error'])
    model.fit(X_train1, y_train1, epochs=100)
    lstmModel = model.summary()
    # train predictions
    train_predictions = model.predict(X_train1).flatten()
    train_results = pd.DataFrame(
        data={'Date': date_train, 'Close Prices': y_train1, 'Train Predictions': train_predictions})
    # train_results

    # test predictions
    test_predictions = model.predict(X_test1).flatten()
    test_results = pd.DataFrame(
        data={'Date': date_test, 'Close Prices': y_test1, 'LSTM Predictions': test_predictions})
    # test_results

    # evaluate model
    mse = mse_eval(test_results['Close Prices'],
                   test_results['LSTM Predictions'])
    mape = mape_eval(test_results['Close Prices'],
                     test_results['LSTM Predictions'])
    print(mse)
    print(mape)

    return test_results, mse, mape


global results
results, lstmMse, lstmMape = evaluate_lstm_model(trainData*.01)

# ARIMA MODEL
# split data


@st.cache
def evaluate_arima_model(df, trainData):
    global arimamodsum
    row = int(len(df)*(trainData*.01))  # 80% testing
    trainingData = list(df[0:row]['Close'])
    testingData = list(df[row:]['Close'])
    predictions = []
    nObservations = len(testingData)

    for i in range(nObservations):
        model = ARIMA(trainingData, order=(pValue, dValue, qValue))  # p,d,q
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = list(output[0])[0]
        predictions.append(yhat)
        actualTestValue = testingData[i]
        trainingData.append(actualTestValue)

    arimamodsum = model_fit.summary()

    # st.write(predictions)
    testingSet = pd.DataFrame(testingData)
    testingSet['ARIMApredictions'] = predictions
    testingSet.columns = ['Close Prices', 'ARIMA Predictions']
    results["ARIMA Predictions"] = testingSet["ARIMA Predictions"]
    MSE = mean_squared_error(testingData, predictions)
    MAPE = mean_absolute_percentage_error(testingData, predictions)

    return MSE, MAPE


# plot all results
arimaMSE, arimaMAPE = evaluate_arima_model(df, trainData)

# plot orig price and predicted price
fig = px.line(results, x=results["Date"], y=["Close Prices", "ARIMA Predictions", "LSTM Predictions"],
              title="BOTH PREDICTED BRENT CRUDE OIL PRICES", width=1000)
st.plotly_chart(fig, use_container_width=True)


details = st.checkbox('Details')
if details:
    st.write(arimamodsum)
    # st.write(lstmModel)


# ACCURACY METRICS
accTable = pd.DataFrame()
accTable['ARIMA-MAPE'] = [arimaMSE]
accTable['ARIMA-MSE'] = [arimaMAPE]
accTable['LSTM-MAPE'] = [lstmMape]
accTable['LSTM-MSE'] = [lstmMse]

# accuracy metrics
st.header("Accuracy Metrics")

st.table(accTable)
