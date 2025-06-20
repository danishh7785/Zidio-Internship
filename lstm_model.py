# lstm_model.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# -----------------------------
# STEP 1: Load and Prepare Data
# -----------------------------
df = pd.read_csv(r"C:\Users\danis\OneDrive\Desktop\Project\cleaned_stock_data.csv", parse_dates=["Date"])
df.set_index("Date", inplace=True)
df = df.asfreq("D")
df["Close"] = df["Close"].interpolate(method='linear')

# -----------------------------
# STEP 2: Normalize and Window the Data
# -----------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[["Close"]])

# Create windowed sequences (X) and targets (y)
window_size = 60
X, y = [], []

for i in range(window_size, len(scaled_data)):
    X.append(scaled_data[i-window_size:i])
    y.append(scaled_data[i])

X, y = np.array(X), np.array(y)

# Reshape for LSTM input (samples, timesteps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

# -----------------------------
# STEP 3: Build the LSTM Model
# -----------------------------
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=10, batch_size=32)

# -----------------------------
# STEP 4: Forecast Next 30 Days
# -----------------------------
# Start with the last 60 days
last_60 = scaled_data[-window_size:]
forecast_input = last_60.reshape(1, window_size, 1)

predictions = []

for _ in range(30):
    next_pred = model.predict(forecast_input, verbose=0)
    predictions.append(next_pred[0, 0])
    next_pred_reshaped = np.reshape(next_pred, (1, 1, 1))  # reshape to (1, 1, 1)
    forecast_input = np.append(forecast_input[:, 1:, :], next_pred_reshaped, axis=1)

# Inverse scale
forecast_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Create date index for forecast
forecast_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=30)

# -----------------------------
# STEP 5: Plot Results
# -----------------------------
plt.figure(figsize=(12,5))
plt.plot(df["Close"], label="Actual")
plt.plot(forecast_dates, forecast_prices, label="LSTM Forecast", color="green")
plt.title("LSTM Forecast (Next 30 Days)")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
