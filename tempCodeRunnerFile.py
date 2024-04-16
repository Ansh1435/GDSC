from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

start = '2010-01-01'
end = '2019-12-31'

model = load_model('stock_prediction.h5')

@app.route('/')
def home():
    return "Welcome to Stock Price Prediction API!"

@app.route('/predict_stock', methods=['POST'])
def predict_stock():
    data = request.get_json()
    stock_ticker = data['stock_ticker']
    
    df = yf.download(stock_ticker, start=start, end=end)
    data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)]) 
    data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

    scaler = MinMaxScaler(feature_range=(0,1))
    data_training_array = scaler.fit_transform(data_training)

    past_100_days = data_training.tail(100)
    final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
    input_data = scaler.fit_transform(final_df)

    x_test = []
    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100:i])
    x_test = np.array(x_test)
    y_predicted = model.predict(x_test)
    
    scaler.scale_
    scale = 1/scaler.scale_[0]
    y_predicted = y_predicted*scale
    
    prediction_dates = df.tail(len(y_predicted)).index
    predicted_prices = pd.Series(y_predicted.flatten(), index=prediction_dates)

    response = {
        "stock_ticker": stock_ticker,
        "predicted_prices": predicted_prices.to_dict()
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
