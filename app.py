from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

start = '2010-01-01'
end = '2019-12-31'

# Load the pre-trained model
model = load_model('stock_prediction.h5')

@app.route('/api/stock-data', methods=['GET'])
def get_stock_data():
    ticker = request.args.get('ticker', 'AAPL')
    df = yf.download(ticker, start=start, end=end)
    stock_data = df.to_json(orient='records')
    return stock_data

@app.route('/api/predictions', methods=['POST'])
def get_predictions():
    ticker = request.json.get('ticker', 'AAPL')
    df = yf.download(ticker, start=start, end=end)
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
    scaler.scale_
    scale = 1/scaler.scale_[0]
    y_predicted = y_predicted*scale
    y_test = y_test*scale

    predictions = {
        'actual_price': y_test.tolist(),
        'predicted_price': y_predicted.tolist()
    }
    return jsonify(predictions)

@app.route('/')
def index():
    # Serve your React app here
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
