from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

import datetime

app = Flask(__name__)

# Load the trained model (Assuming you saved it as 'stock_model.h5')
model = load_model('stock_model.h5')

# Function to fetch stock data and prepare the dataset
def prepare_data(stock_symbol):
    # Fetch stock data using yfinance
    data = yf.download(stock_symbol, period="5y", interval="1d")  # Get 5 years of data
    data = data[['Close']]  # Use only 'Close' price

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Prepare input (X) and output (Y)
    X = []
    Y = []
    time_step = 60
    for i in range(len(scaled_data) - time_step):
        X.append(scaled_data[i:i + time_step, 0])  # Input (last 60 days)
        Y.append(scaled_data[i + time_step, 0])    # Output (next day's price)
    
    X = np.array(X)
    Y = np.array(Y)

    # Reshape input data for LSTM (samples, time steps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, Y, scaler

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        stock_symbol = request.form['stock_symbol']
        
        try:
            # Prepare the data
            X, Y, scaler = prepare_data(stock_symbol)

            # Predict the stock price for the last day in the dataset
            prediction = model.predict(X[-1].reshape(1, 60, 1))  # Predict the next day's price
            prediction = scaler.inverse_transform(prediction.reshape(-1, 1))  # Convert back to original scale

            # Get the date of the prediction
            prediction_date = (datetime.datetime.now() + datetime.timedelta(days=1)).strftime('%Y-%m-%d')

            return render_template('index.html', prediction=prediction[0][0], prediction_date=prediction_date)
        
        except Exception as e:
            return render_template('index.html', error_message=str(e))

if __name__ == "__main__":
    app.run(debug=True)
