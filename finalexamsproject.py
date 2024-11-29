import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from prophet import Prophet

# Streamlit App Title and Description
st.title("Time Series Data Analysis and Forecasting")
st.markdown("""
This Streamlit app performs time series analysis on goods' pricing data. It includes:
- Viewing raw data
- Visualizing trends, histograms, and scatter plots
- Handling missing values
- Forecasting future prices
""")

# Sidebar Navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Select Section", [
    "Upload Data", "View Data", "Visualizations", "Missing Values", "Forecasting"
])

# File Upload
if section == "Upload Data":
    st.header("Upload Data")
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
    else:
        st.info("Please upload a CSV file to proceed.")

# Load and Display Data
if section == "View Data" and uploaded_file:
    st.header("View Data")
    st.write("Here is the data you uploaded:")
    st.dataframe(data)
    st.markdown(f"**Shape of the Data:** {data.shape}")

    st.markdown("### Column Details:")
    st.write(data.dtypes)

# Visualizations Section
if section == "Visualizations" and uploaded_file:
    st.header("Visualizations")

    # Time Series Plot
    st.markdown("### Time Series Plot")
    if 'date' in data.columns and 'price' in data.columns:
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)

        plt.figure(figsize=(10, 5))
        plt.plot(data.index, data['price'], label="Price")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.title("Time Series of Prices")
        plt.legend()
        st.pyplot(plt)

    # Histogram
    st.markdown("### Histogram of Prices")
    plt.figure(figsize=(8, 5))
    sns.histplot(data['price'], bins=20, kde=True)
    plt.title("Price Distribution")
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    st.pyplot(plt)

    # Scatter Plot
    st.markdown("### Scatter Plot")
    columns = data.columns.tolist()
    x_col = st.selectbox("Select X-axis Column", columns)
    y_col = st.selectbox("Select Y-axis Column", columns)
    if x_col and y_col:
        plt.figure(figsize=(8, 5))
        sns.scatterplot(x=data[x_col], y=data[y_col])
        plt.title(f"Scatter Plot: {x_col} vs {y_col}")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        st.pyplot(plt)

# Handle Missing Values
if section == "Missing Values" and uploaded_file:
    st.header("Missing Values")
    st.write("Checking for missing values...")
    missing = data.isnull().sum()
    st.write(missing)

    st.markdown("### Handling Missing Values")
    fill_method = st.selectbox("Select a Method", ["Forward Fill", "Backward Fill", "Drop Rows"])
    if fill_method == "Forward Fill":
        data.fillna(method='ffill', inplace=True)
    elif fill_method == "Backward Fill":
        data.fillna(method='bfill', inplace=True)
    elif fill_method == "Drop Rows":
        data.dropna(inplace=True)

    st.write("Missing values handled successfully!")
    st.write("Updated Data:")
    st.dataframe(data)

# Forecasting Section
if section == "Forecasting" and uploaded_file:
    st.header("Forecasting")

    st.markdown("### Linear Regression Forecast")
    if 'date' in data.columns and 'price' in data.columns:
        last_12 = data['price'].dropna().iloc[-12:]
        X = pd.DataFrame(range(len(last_12)))
        y = last_12.values

        # Linear Regression
        model = LinearRegression()
        model.fit(X, y)
        future_X = pd.DataFrame(range(len(last_12), len(last_12) + 6))
        forecast = model.predict(future_X)

        # Plot Linear Regression Forecast
        future_dates = pd.date_range(start=data.index[-1], periods=7, freq='M')[1:]
        plt.figure(figsize=(10, 5))
        plt.plot(last_12.index, last_12.values, label="Last 12 Months")
        plt.plot(future_dates, forecast, label="Linear Regression Forecast")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        st.pyplot(plt)

    st.markdown("### Simple Exponential Smoothing Forecast")
    ses_model = SimpleExpSmoothing(data['price'].dropna()).fit(smoothing_level=0.2, optimized=False)
    ses_forecast = ses_model.forecast(6)
    future_dates = pd.date_range(start=data.index[-1], periods=7, freq='M')[1:]

    # Plot SES Forecast
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data['price'], label="Actual Prices")
    plt.plot(future_dates, ses_forecast, label="SES Forecast")
    plt.legend()
    st.pyplot(plt)

    st.markdown("### Prophet Forecast")
    prophet_data = data[['date', 'price']].rename(columns={"date": "ds", "price": "y"})
    prophet_model = Prophet()
    prophet_model.fit(prophet_data)
    future = prophet_model.make_future_dataframe(periods=6, freq='M')
    prophet_forecast = prophet_model.predict(future)

    # Plot Prophet Forecast
    prophet_model.plot(prophet_forecast)
    st.pyplot(plt)
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from prophet import Prophet

# Title of the App
st.title("Time Series Analysis and Forecasting")

# Markdown Description
st.markdown("""
### Introduction
This app performs time series analysis and forecasting on various goods' pricing data.
You can view data, visualize trends, handle missing values, and forecast future prices using different methods.
""")

# Sidebar Navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Select Section", ["Data Loading", "Data Analysis", "Missing Values", "Visualizations", "Forecasting"])

# Load Data
@st.cache
def load_data():
    # Replace with your actual data loading process
    # For now, we simulate data as an example
    data = pd.DataFrame({
        'mp_year': [2020, 2020, 2021],
        'mp_month': [1, 2, 1],
        'mp_price': [100, 110, 120],
    })
    data['date'] = pd.to_datetime(data['mp_year'].astype(str) + '-' + data['mp_month'].astype(str) + '-01')
    data.set_index('date', inplace=True)
    return data

data = load_data()

if section == "Data Loading":
    st.header("Data Loading")
    st.markdown("The data consists of pricing information for different goods over time.")
    st.dataframe(data)

elif section == "Data Analysis":
    st.header("Data Analysis")
    st.markdown("### Data Overview")
    st.write("Shape of the DataFrame:", data.shape)
    st.write("Data Types:")
    st.write(data.dtypes)

    st.markdown("### Summary Statistics")
    st.write(data.describe())

elif section == "Missing Values":
    st.header("Missing Values")
    st.markdown("### Handling Missing Data")
    st.write("Missing values before filling:")
    st.write(data.isnull().sum())

    # Forward fill as an example
    data = data.ffill()

    st.write("Missing values after forward filling:")
    st.write(data.isnull().sum())

elif section == "Visualizations":
    st.header("Visualizations")
    st.markdown("### Time Series Plot")
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data['mp_price'], label='Price')
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("Time Series of Prices")
    plt.legend()
    st.pyplot(plt)

elif section == "Forecasting":
    st.header("Forecasting")
    st.markdown("### Linear Regression Forecasting")
    last_12_months = data['mp_price'][-12:]
    X = range(len(last_12_months))
    y = last_12_months.values
    model = LinearRegression()
    model.fit(pd.DataFrame(X), y)
    future_X = pd.DataFrame(range(len(last_12_months), len(last_12_months) + 6))
    forecast = model.predict(future_X)

    # Plot Linear Regression Forecast
    plt.figure(figsize=(10, 5))
    plt.plot(last_12_months.index, last_12_months, label='Last 12 Months')
    future_dates = pd.date_range(start=last_12_months.index[-1], periods=7, freq='M')[1:]
    plt.plot(future_dates, forecast, label='Forecast')
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(plt)

    st.markdown("### Simple Exponential Smoothing Forecasting")
    ses_model = SimpleExpSmoothing(data['mp_price']).fit(smoothing_level=0.2, optimized=False)
    ses_forecast = ses_model.forecast(6)
    future_dates = pd.date_range(start=data.index[-1], periods=7, freq='M')[1:]
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data['mp_price'], label='Actual Prices')
    plt.plot(future_dates, ses_forecast, label='SES Forecast')
    plt.legend()
    st.pyplot(plt)

    st.markdown("### Prophet Forecasting")
    prophet_data = data.reset_index().rename(columns={'date': 'ds', 'mp_price': 'y'})
    prophet_model = Prophet()
    prophet_model.fit(prophet_data)
    future = prophet_model.make_future_dataframe(periods=6, freq='M')
    prophet_forecast = prophet_model.predict(future)
    prophet_model.plot(prophet_forecast)
    st.pyplot(plt)
