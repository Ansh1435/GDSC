import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Define start and end dates
start = '2010-01-01'
end = '2019-12-31'

# Retrieve data using yfinance
df = yf.download('AAPL', start=start, end=end)

# Display the first few rows of the DataFrame
ma100 = df['Close'].rolling(window=100).mean()
ma200 = df['Close'].rolling(window=200).mean()
plt.figure(figsize=(12, 6))
plt.plot(df.Close)
plt.title('AAPL Closing Prices from 2010 to 2019')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.grid(True)
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

# ml model
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from keras.layers import Input

model = Sequential()
model.add(Input(shape=(x_train.shape[1], x_train.shape[2])))
model.add(LSTM(units=50, activation='relu', return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=60, activation='relu', return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(units=80, activation='relu', return_sequences=True))
model.add(Dropout(0.4))

model.add(LSTM(units=120, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=50)

model.save('stock_prediction.h5')

# Prepare the test data
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)
x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)

# Make predictions
y_predicted = model.predict(x_test)
scale_factor = 1 / scaler.scale_[0]  # Inverse transform scale factor
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(y_test, color='b', label='Real AAPL Stock Price')
plt.plot(y_predicted, color='r', label='Predicted AAPL Stock Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
