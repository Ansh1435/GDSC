import React, { useState } from 'react';
import axios from 'axios';

function StockPrediction() {
  const [stockTicker, setStockTicker] = useState('');
  const [predictions, setPredictions] = useState({});

  const predictStock = async () => {
      try {
          const response = await axios.get(`http://localhost:5000/predict_stock/${stockTicker}`);
          setPredictions(response.data.predicted_prices);
      } catch (error) {
          console.error('Error fetching predictions:', error);
      }
  };

  return (
      <div>
          <h2>Stock Price Prediction</h2>
          <input type="text" placeholder="Enter Stock Ticker" value={stockTicker} onChange={(e) => setStockTicker(e.target.value)} />
          <button onClick={predictStock}>Predict</button>
          <h3>Predictions:</h3>
          <ul>
              {Object.entries(predictions).map(([date, price]) => (
                  <li key={date}>{date}: {price}</li>
              ))}
          </ul>
      </div>
  );
}


export default StockPrediction;
