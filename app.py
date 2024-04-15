from flask import Flask, jsonify, render_template, send_from_directory
from flask_cors import CORS
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os

app = Flask(__name__, static_folder='static')
CORS(app)

start = '2010-01-01'
end = '2019-12-31'
model = load_model('stock_prediction.h5')

@st.cache
def run_streamlit(user_input):
    # Your Streamlit app code here...
    df = yf.download(user_input, start=start, end=end)

    st.subheader('Data from 2010 - 2019')
    st.write(df.describe())

    st.subheader('Closing Price vs Time chart')
    fig = plt.figure(figsize=(12,6))
    plt.plot(df.Close)
    st.pyplot(fig)

    ma100 = df.Close.rolling(100).mean()
    st.subheader('Closing Price vs Time chart with 100MA')
    fig = plt.figure(figsize = (12,6))
    plt.plot(ma100)
    plt.plot(df.Close)
    st.pyplot(fig)

    ma100 = df.Close.rolling(100).mean()
    ma200 = df.Close.rolling(200).mean()
    st.subheader('Closing Price vs Time chart with 100MA & 200MA')
    fig = plt.figure(figsize = (12,6))
    plt.plot(ma100)
    plt.plot(ma200)
    plt.plot(df.Close)
    st.pyplot(fig)

    data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)]) 
    data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])
    
    scaler = MinMaxScaler(feature_range=(0,1))
    data_training_array = scaler.fit_transform(data_training)
    
    past_100_days = data_training.tail(100)
    final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
    input_data = scaler.fit_transform(final_df)
    
    x_test = []
    y_test = []
    
    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100:i])
        y_test.append(input_data[i, 0])
    x_test, y_test = np.array(x_test), np.array(y_test)
    y_predicted = model.predict(x_test)
    scale = 1/scaler.scale_[0]
    y_predicted = y_predicted*scale
    y_test = y_test*scale

    st.subheader('Predictions vs Real Price')
    fig3 = plt.figure(figsize=(12,6))
    plt.plot(y_test, 'b', label = 'Actual Price')
    plt.plot(y_predicted, 'r', label = 'Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig3)

@app.route('/')
def home():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/predictions/<stock_ticker>')
def get_predictions(stock_ticker):
    run_streamlit(stock_ticker)
    return 'Done'

# Route to serve static assets from the React build folder
@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory(os.path.join(app.static_folder, 'react'), path)

if __name__ == '__main__':
    app.run(debug=True)

