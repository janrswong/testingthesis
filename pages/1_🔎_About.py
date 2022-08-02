import streamlit as st
# from PIL import Image
import base64

# function play gif
def gif(location):
    """### gif from local file"""
    file_ = open(location, "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

#     return (st.markdown(
#         f'<img src="data:image/gif;base64,{data_url}" alt="instructions gif">',
#         unsafe_allow_html=True,
#     ))
    return (st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="instructions gif" width="100%">',
        unsafe_allow_html=True,
    ))

# page expands to full width
st.set_page_config(page_title="About", layout='wide')

st.title('About the Research')

# a. how the thing works
st.header('How It Works')
st.markdown('<strong>Welcome!</strong> This is a fully interactive, multi-page web app through the Python library Streamlit that allows users to explore the same models used in the study. Aside from learning about study findings, play with parameters, create your own models, conduct your own comparisions and make your own analyses! Read further to learn how to use the <em>Explore</em> and <em>Make Your Own Model</em> tabs.', unsafe_allow_html=True)

# a.1 How to use the explore tab
st.markdown('### How to use the <a href="/Explore" target="_self" style="text-decoration:none; color: tomato;">📈 Explore</a>' ' tab', unsafe_allow_html=True)
# st.subheader("How to use the 'Explore' tab")
st.markdown('<p> The Explore tab allows you to <strong>select different models</strong> based on time intervals and test-train split data. <br> </p>', unsafe_allow_html=True)

st.markdown(
    '<p> Use the slider to select whether or not you would like to see monthly, weekly, daily, or quarterly data being used. </p>', unsafe_allow_html=True)
gif('assets/images/Slider.gif')



st.markdown('<p> Then, use the dropdown to select your split. </p>',
            unsafe_allow_html=True)
gif('assets/images/Split.gif')
# split = Image.open('assets/images/Split.jpg')
# st.image(split, caption='Explore Split')

st.markdown('<p>  Afterwards, you can use the select tool to compare as many ARIMA models as you like, either against each other or against an LSTM model! </p>', unsafe_allow_html=True)
# select = Image.open('assets/images/Select.jpg')
# st.image(select, caption='Explore Select')
gif('assets/images/Select.gif')




# a.2 How to use the Make a model tab
st.markdown('### How to use the <a href="Make_a_Model" target="_self" style="text-decoration:none; color: tomato;">📊 Make a Model</a>' ' tab', unsafe_allow_html=True)
st.markdown('<p> The Make a Model tab lets you <strong>create your own model</strong>! Choose the different parameters and test-train splits to your liking for an interactive experience with our time series models. <br> </p>', unsafe_allow_html=True)


# b. snippets of the paper
st.header('Documentation')
st.markdown('<p> The study entitled: <em>Predicting the Investment Landscape: A Comparative Analysis on the Autoregressive Integrated Moving Average (ARIMA) Model and the Long Short Term Memory (LSTM) Model Through Benchmark Crude Oil Prices</em> revolves around taking Brent crude oil prices from 2007-2022 and using them to compare two popular forcasting methods. <br> <br>Due to the lack of comprehensive and updated quantitative studies that consider the Philippine foreign dependency, precious, new, and volatile financial assets such as the energy and crude oil sector. The need to provide economists and finance researchers with new, accurate prediction methods has become increasingly important. Additionally, inconvenienced international and local investors might be un-incentivized to invest in the industry, making these assets less mobile in the PH and other countries.<br><br> Brent standard crude oil has become particularly of interest due to its influence on the international oil market. In conjunction with other pricing standards like the OMAN and WTIthis pricing standard serves as a benchmark for pricing oil for countries all over the world, especially in the Philippines. The Brent standard also has a well-documented time-series record, making it a desirable focus for this study. </p>', unsafe_allow_html=True)

# sample text
st.markdown('Here are some code snippets!')
st.markdown('Here are some code snippets that illustrate the saving and evaluation of the ARIMA and LSTM models in Google Colab')

# sample add code snippet

code = '''# evaluate combinations of p, d and q values for an ARIMA model
# prepare training dataset
train_size = int(len(X) * split)
train, test = X[0:train_size], X[train_size:]
history = [x for x in train]
predset = pd.DataFrame(test)

# make predictions
predictions = list()
for t in range(len(test)):
     model = ARIMA(history, order=arima_order)
     model_fit = model.fit()
     yhat = model_fit.forecast()[0]
     predictions.append(*yhat)
     history.append(test[t])'''
st.code(code, language='python')
code = '''p_values = range(0, 3)
d_values = range(0, 3)
q_values = range(0, 2)
split_values = [0.8, 0.5, 0.6]
warnings.filterwarnings("ignore")
df1 = evaluate_models(df.Close, p_values, d_values, q_values,split_values)'''
st.code(code, language='python')
# LSTM
code = '''# LSTM Model
# preprocessing
    date_train, date_val, date_test = df.index[:int(df.shape[0]*split)],df.index[int(df.shape[0]*split):int(df.shape[0]*val_split)], df.index[int(df.shape[0]*val_split)+WINDOW_SIZE:]
    X_train1, y_train1 = X1[:int(df.shape[0]*split)], y1[:int(df.shape[0]*split)]
    X_val, y_val = X1[int(df.shape[0]*split):int(df.shape[0]*val_split)], y1[int(df.shape[0]*split):int(df.shape[0]*val_split)]
    X_test1, y_test1 = X1[int(df.shape[0]*val_split):], y1[int(df.shape[0]*val_split):]

    # X_train1.shape, y_train1.shape, X_test1.shape, y_test1.shape

    # lstm model
    model = Sequential([layers.Input((3,1)),layers.LSTM(64),layers.Dense(32, activation='relu'),layers.Dense(32, activation='relu'), layers.Dense(1)])
    cp1 = ModelCheckpoint('model1/', save_best_only=True)
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=['mean_absolute_percentage_error'])
    model.fit(X_train1, y_train1,validation_data=(X_val, y_val), epochs=100, callbacks=[cp1])
    # model.summary()
'''
st.code(code, language='python')


# code = '''def hello():
#      print("Hello, Streamlit!")'''
# st.code(code, language='python')

# sample add latex or formulas
st.latex(r'''
     a + ar + a r^2 + a r^3 + \cdots + a r^{n-1} =
     \sum_{k=0}^{n-1} ar^k =
     a \left(\frac{1-r^{n}}{1-r}\right)
     ''')

# c. about us creators
st.header('About the Authors')
st.markdown('The authors, Janna Rizza Wong, Stephanie Lorraine Ignas, and Alyanna Angela Castillon, are all university seniors taking Computer Science.')



