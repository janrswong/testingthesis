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
# from tensorflow.keras import layers
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import ModelCheckpoint
from st_aggrid import AgGrid

# page expands to full width
st.set_page_config(page_title="LSTM vs ARIMA", layout='wide')

# PAGE LAYOUT
# heading
st.title("Crude Oil Benchmark Stock Price Prediction LSTM and ARIMA Models")
st.subheader("""© Castillon, Ignas, Wong""")


# select time interval
interv = st.select_slider('Select Time Series Data Interval for Prediction', options=[
                          'Weekly', 'Monthly', 'Quarterly', 'Yearly'])

# st.write(interv[0])

# Function to convert time series to interval


def getInterval(argument):
    switcher = {
        "W": "1wk",
        "M": "1mo",
        "Q": "3mo",
        "Y": "1d"
    }
    return switcher.get(argument, "1d")


# show raw data
st.header("Raw Data")
# using button
# if st.button('Press to see Brent Crude Oil Raw Data'):
df = yf.download('BZ=F', interval=getInterval(interv[0]))
st.dataframe(df.head())

# TODO: standard deviation

# AgGrid(df)

# accuracy metrics
st.header("Accuracy Metric Comparison")

# LSTM METRICS
st.write("LSTM Metrics")
readfile = pd.read_csv('LSTM.csv')
readfile.drop("Unnamed: 0", axis=1)
AgGrid(readfile, key='LSTMMetric')

# ARIMA METRICS
st.write("ARIMA Metrics")
intervals = st.selectbox(
    "Select Interval:", ('Weekly', 'Monthly', 'Quarterly', 'Daily'))

if intervals == 'Weekly':
    file = pd.read_csv('ARIMAMetrics/ARIMA-WEEKLY.csv')
    file.drop("Unnamed: 0", axis=1)
    AgGrid(file, key='weeklyMetric')

elif intervals == 'Monthly':
    file = pd.read_csv('ARIMAMetrics/ARIMA-MONTHLY.csv')
    file.drop("Unnamed: 0", axis=1)
    AgGrid(file, key='monthlyMetric')

elif intervals == 'Quarterly':
    file = pd.read_csv('ARIMAMetrics/ARIMA-QUARTERLY.csv')
    file.drop("Unnamed: 0", axis=1)
    AgGrid(file, key='quarterlyMetric')

elif intervals == 'Daily':
    file = pd.read_csv('ARIMAMetrics/ARIMA-DAILY.csv')
    file.drop("Unnamed: 0", axis=1)
    AgGrid(file, key='dailyMetric')

# MODEL OUTPUT TABLE
st.header("Model Output (Close Prices vs. Predicted Prices)")

interval = st.selectbox("Select Interval:", ('Weekly',
                        'Monthly', 'Quarterly', 'Daily'), key='bestmodels')

if interval == 'Weekly':
    file = pd.read_csv('bestWeekly.csv')
   
    AgGrid(file, key='weeklycombined')

    # Visualization
    st.header("Visualization")
    fig = px.line(file, x=file["Date"], y=["Close Prices", "ARIMA_50.0_(0, 1, 1)_Predictions", "LSTM_80.0_Predictions"], title="BOTH PREDICTED BRENT CRUDE OIL PRICES", width=1000)
    st.plotly_chart(fig, use_container_width=True)


elif interval == 'Monthly':
    file = pd.read_csv('bestMonthly.csv')
    
    AgGrid(file, key='monthlyCombined')
    # Visualization
    st.header("Visualization")
    fig = px.line(file, x=file["Date"], y=["Close Prices", "ARIMA_50.0_(0, 1, 0)_Predictions", #find file
                  "LSTM_65.0_Predictions"], title="BOTH PREDICTED BRENT CRUDE OIL PRICES", width=1000)
    st.plotly_chart(fig, use_container_width=True)


elif interval == 'Quarterly':
    file = pd.read_csv('bestQuarterly.csv')
    
    AgGrid(file, key='quarterlyCombined')
    # Visualization
    st.header("Visualization")
    fig = px.line(file, x=file["Date"], y=["Close Prices", "ARIMA_50.0_(0, 1, 0)_Predictions", #find file
                  "LSTM_80.0_Predictions"], title="BOTH PREDICTED BRENT CRUDE OIL PRICES", width=1000)
    st.plotly_chart(fig, use_container_width=True)


elif interval == 'Daily':
    file = pd.read_csv('bestDaily.csv')
    
    AgGrid(file, key='dailyCombined')
     # Visualization
    st.header("Visualization")
    fig = px.line(file, x=file["Date"], y=["Close Prices", "ARIMA_50.0_(0, 1, 1)_Predictions", #find file
                "LSTM_65.0_Predictions"],title="BOTH PREDICTED BRENT CRUDE OIL PRICES", width=1000)
    st.plotly_chart(fig, use_container_width=True)

