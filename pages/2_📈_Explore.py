# TODO: add combined data w selection(explore), add statefulness in selects

import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
# import numpy as np
import plotly.express as px
from st_aggrid import GridOptionsBuilder, AgGrid

# page expands to full width
st.set_page_config(page_title="Explore Models", layout='wide')

# ARIMA
# slider interval
interv = st.select_slider('Select Time Series Data Interval for Prediction', options=[
                          'Weekly', 'Monthly', 'Quarterly', 'Daily'], value='Weekly')

# dropdown 50 60 80
st.write("Select Split")
intervals = st.selectbox(
    "Select Interval:", ('80', '60', '50'))

# read file from select interval and dropdown split


def get_location(interv, intervals):
    location = 'Explore/ARIMA/' + interv + '/' + intervals + '.csv'
    return location


location = get_location(interv, intervals)

# pagination function for aggrid


def pagination(df):
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_pagination(paginationAutoPageSize=True)
    return gb.build()


# read file
file = pd.read_csv(get_location(interv, intervals))
page = pagination(file)
file.drop("Unnamed: 0", axis=1, inplace=True)
# show table
# AgGrid(file, width='100%', theme='streamlit',
#        fit_columns_on_grid_load=True, gridOptions=page)

# select columns
columns = file.columns.to_list()
# st.write(columns)
selectedCols= st.multiselect("Select models",columns)
df = file[selectedCols]
st.dataframe(df)

# LSTM
