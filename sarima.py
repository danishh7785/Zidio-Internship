# sarima_model.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# -----------------------------
# STEP 1: Load and Prepare Data
# -----------------------------
df = pd.read_csv(r"C:\Users\danis\OneDrive\Desktop\Project\cleaned_stock_data.csv", parse_dates=["Date"])
df.set_index("Date", inplace=True)
df = df.asfreq("D")
df["Close"] = df["Close"].interpolate(method='linear')

# -----------------------------
# STEP 2: ADF Stationarity Test
# -----------------------------
def adf_test(series, title=''):
    print(f"\n--- ADF Test: {title} ---")
    result = adfuller(series.dropna())
    print(f"ADF Statistic : {result[0]}")
    print(f"p-value       : {result[1]}")
    if result[1] <= 0.05:
        print("✅ The series is stationary.")
    else:
        print("❌ The series is NOT stationary.")

adf_test(df["Close"], title="Original Series")

# -----------------------------
# STEP 3: Differencing if Needed
# -----------------------------
df["Close_diff"] = df["Close"].diff()
adf_test(df["Close_diff"], title="1st Differenced Series")

# -----------------------------
# STEP 4: Fit SARIMA Model
# -----------------------------
# SARIMA(p,d,q)(P,D,Q,S)
# We’ll use seasonal_order=(1,1,1,12) for monthly seasonality
model = SARIMAX(df["Close"], order=(1,1,1), seasonal_order=(1,1,1,12))
model_fit = model.fit()
print("\n--- SARIMA Model Summary ---")
print(model_fit.summary())

# -----------------------------
# STEP 5: Forecast Future Values
# -----------------------------
forecast_days = 30
forecast = model_fit.forecast(steps=forecast_days)
forecast_index = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)

# -----------------------------
# STEP 6: Plot Actual vs Forecast
# -----------------------------
plt.figure(figsize=(12, 5))
plt.plot(df["Close"], label="Actual Close Price")
plt.plot(forecast_index, forecast, label="SARIMA Forecast (Next 30 Days)", color='orange')
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("SARIMA Forecast")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# STEP 7: Evaluate Model (Optional)
# -----------------------------
if len(df) >= 60:
    actual = df["Close"][-30:]
    predicted = model_fit.predict(start=len(df)-30, end=len(df)-1)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    print(f"\nRMSE on last 30 days: {rmse:.2f}")
