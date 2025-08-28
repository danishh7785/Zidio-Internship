# 📈 Stock Market Time Series Forecasting

This project focuses on analyzing and forecasting stock prices using **time series models** such as **ARIMA, SARIMA, Prophet, and LSTM**.  
The goal is to identify **trends, seasonality, and patterns** in historical stock market data and evaluate multiple forecasting models for both **short-term and long-term predictions**.

---

## 📂 Project Structure

stock-market-forecasting/
│── data/
│   ├── aapl_stock.csv              # Raw stock dataset
│   ├── cleaned_stock_data.csv      # Preprocessed dataset
│
│── notebooks/
│   ├── EDA.ipynb                   # Data exploration & preprocessing
│   ├── EDA_Visualizations.ipynb    # Data visualizations
│   ├── data_cleaning.ipynb         # Cleaning steps
│   ├── webscraping.ipynb           # Web scraping for stock data
│
│── src/
│   ├── arima.py                    # ARIMA model
│   ├── sarima.py                   # SARIMA model
│   ├── lstm_model.py               # LSTM model
│
│── dashboards/
│   ├── simple_dashboard.html       # Simple dashboard for results
│   ├── sarima_dashboard.html       # SARIMA visualization dashboard
│   ├── lstm_dashboard.html         # LSTM visualization dashboard
│
│── results/
│   ├── plots/                      # Forecast and comparison plots
│   └── metrics/                    # Evaluation metrics (future)
│
│── README.md                       # Project documentation
│── requirements.txt                # Dependencies (to be added)
│── LICENSE                         # (optional: MIT/Apache)
│── .gitignore                      # Ignore unnecessary files




## 🛠️ Methods Used
1. **Exploratory Data Analysis (EDA)**  
   - Visualization of stock trends, returns, and seasonality  
   - Stationarity checks (ADF Test)  

2. **Models Implemented**
   - 📌 **ARIMA** – for short-term linear forecasting  
   - 📌 **SARIMA** – capturing seasonality in time series  
   - 📌 **Prophet (by Meta)** – robust to holidays & trends  
   - 📌 **LSTM (Deep Learning)** – captures long-term dependencies  
