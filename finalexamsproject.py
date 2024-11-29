import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from prophet import Prophet
import pmdarima as pm

# Set Seaborn style
sns.set_style("whitegrid")

# Function to parse the date
def parse_date(df):
    df['date'] = pd.to_datetime(df['mp_year'].astype(str) + '-' + df['mp_month'].astype(str) + '-01')
    return df

# Load CSV files
data_path = '/path/to/your/data/'  # Change this to the directory path
csv_files = [
    'Beans (dry).csv', 'Cassava.csv', 'Chili (red).csv', 'Maize.csv',
    'Oranges (big size).csv', 'Peas (fresh).csv', 'Potatoes (Irish).csv',
    'Sorghum.csv', 'Tomatoes.csv'
]

# Read and preprocess each CSV
dataframes = {}
for file in csv_files:
    file_path = data_path + file
    df = pd.read_csv(file_path)
    if 'mp_year' in df.columns and 'mp_month' in df.columns:
        df = parse_date(df)
        df.set_index('date', inplace=True)
    dataframes[file] = df

# Combine all DataFrames
combined_df = pd.concat(dataframes.values(), keys=dataframes.keys())

# Fill missing values using forward fill
for name, df in dataframes.items():
    dataframes[name] = df.fillna(method='ffill')

# Data Overview
st.write("Combined DataFrame Shape:", combined_df.shape)
for name, df in dataframes.items():
    st.write(f"DataFrame: {name}, Shape: {df.shape}")
    st.write(df.dtypes)

# Visualize individual time series
for name, df in dataframes.items():
    plt.figure(figsize=(12, 6))
    if 'mp_price' in df.columns:
        sns.lineplot(x=df.index, y=df['mp_price'], label=name.split('.')[0])
    plt.title(f"Time Series of {name}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt)

# Correlation Analysis
price_data = [df['mp_price'].reset_index(drop=True) for df in dataframes.values()]
price_df = pd.DataFrame(price_data).transpose()
price_df.columns = list(dataframes.keys())
correlation_matrix = price_df.corr()

# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Goods Prices')
st.pyplot(plt)

# Moving Average for Beans (dry)
df_beans = dataframes['Beans (dry).csv']
beans_prices = df_beans['mp_price']
window_sizes = [3, 6, 12]

for window in window_sizes:
    beans_prices_ma = beans_prices.rolling(window=window).mean()
    plt.figure(figsize=(12, 6))
    plt.plot(beans_prices, label='Original Prices')
    plt.plot(beans_prices_ma, label=f'{window}-Month Moving Average')
    plt.title(f"Beans Prices and {window}-Month Moving Average")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(plt)

# ARIMA Forecast
arima_model = pm.auto_arima(beans_prices, seasonal=False, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
model = ARIMA(beans_prices, order=arima_model.order)
model_fit = model.fit()
forecast_steps = 6
forecast_arima = model_fit.predict(start=len(beans_prices), end=len(beans_prices) + forecast_steps - 1)
forecast_index = pd.date_range(start=beans_prices.index[-1], periods=forecast_steps + 1, freq='M')[1:]

# Simple Exponential Smoothing
ses_model = SimpleExpSmoothing(beans_prices).fit(smoothing_level=0.2)
ses_forecast = ses_model.forecast(6)

# Prophet Forecast
prophet_df = pd.DataFrame({'ds': beans_prices.index, 'y': beans_prices.values})
prophet_model = Prophet()
prophet_model.fit(prophet_df)
future = prophet_model.make_future_dataframe(periods=6, freq='M')
forecast_prophet = prophet_model.predict(future)

# Plotting Forecasts
plt.figure(figsize=(15, 8))
plt.plot(beans_prices, label='Actual Prices', color='blue')
plt.plot(forecast_index, forecast_arima, label='ARIMA Forecast', color='red')
plt.plot(forecast_prophet['ds'], forecast_prophet['yhat'], label='Prophet Forecast', color='green')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Beans Price Forecasts (ARIMA, Prophet)')
plt.legend()
plt.grid(True)
st.pyplot(plt)
