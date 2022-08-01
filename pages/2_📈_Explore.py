# TODO: add combined data w selection(explore), add statefulness in selects

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
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
selectedCols = st.multiselect("Select models", columns)
df = file[selectedCols]
st.dataframe(df)

# get list of columns
pltcols = list(df)
st.dataframe(pltcols)
st.write(pltcols[0])

collist = list()
collist.append(pltcols)
for cols in collist:
    st.write(cols)

st.write(collist)

# plot selected columns
# fig = px.line(df, x=df['Date'], y=[df['Close Prices'], df['ARIMA_80.0_(2, 0, 0)_Predictions']], title="Prices", width=1000)
# st.plotly_chart(fig, use_container_width=True)

# TODO: Plot df in same window
# working but opens on separate window
fig = go.Figure()
for idx, col in enumerate(df.columns, 0):
    fig.add_trace(go.Scatter(
        x=file['Date'], y=df.iloc[1:, idx], mode='lines', name=col))
fig.show()
# LSTM
