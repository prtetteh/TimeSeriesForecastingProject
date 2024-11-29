import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.linear_model import LinearRegression

# Streamlit app title and markdown header
st.title("Time Series Analysis and Forecasting App")
st.markdown("""
This app explores a time series dataset and provides various analysis and forecasting functionalities.
Upload your CSV file containing time series data to get started.
""")

# File upload section
uploaded_file = st.file_uploader("Upload CSV File", type="csv")

# Function to parse the date column (assuming 'mp_year' and 'mp_month' columns)
def parse_date(df):
    df['date'] = pd.to_datetime(df['mp_year'].astype(str) + '-' + df['mp_month'].astype(str) + '-01')
    return df

# Function to verify date range continuity
def verify_date_range(df):
    if not isinstance(df.index, pd.DatetimeIndex):
        return False, "Error: Index is not a DateTimeIndex."

    start_date = df.index.min()
    end_date = df.index.max()
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')  # Monthly start frequency

    is_continuous = df.index.equals(date_range)

    description = (
        f"Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n"
        f"Continuous: {is_continuous}"
    )

    if not is_continuous:
        missing_dates = date_range.difference(df.index)
        description += f"\nMissing Dates: {missing_dates.tolist()}"

    return is_continuous, description

# Data processing and analysis logic
if uploaded_file is not None:
    # Read the uploaded CSV file
    df = pd.read_csv(uploaded_file)

    # Check if 'mp_year' and 'mp_month' columns exist for date parsing
    if 'mp_year' in df.columns and 'mp_month' in df.columns:
        df = parse_date(df)
        df.set_index('date', inplace=True)  # Set 'date' as the index
    else:
        st.warning("Date parsing columns ('mp_year', 'mp_month') not found. Data assumed to have a DatetimeIndex.")

    # Display basic data information
    st.header("Data Overview")
    st.write("Shape:", df.shape)
    st.write("Column names and data types:", df.dtypes)

    # Verify date range continuity
    is_continuous, description = verify_date_range(df)
    st.write("Date Range:", description)

    # Handle missing values section
    st.header("Missing Values")

    # Option 1: Fill missing values with NaN
    df_with_nan = df.copy()
    df_with_nan = df_with_nan.fillna(value=pd.NA)

    # Option 2: Interpolate missing values (e.g., forward fill)
    df_ffill = df.copy()
    df_ffill = df_ffill.ffill()  # Fill missing values using forward fill

    # Allow user to choose between options
    selected_missing_value_handling = st.selectbox(
        "Missing Value Handling Method",
        options=["Fill with NaN", "Forward Fill (ffill)"]
    )

    if selected_missing_value_handling == "Fill with NaN":
        df_to_use = df_with_nan
    else:
        df_to_use = df_ffill

    # Display missing values after handling
    st.write("Missing values after handling:", df_to_use.isnull().sum())

    # Descriptive statistics section
    st.header("Descriptive Statistics")
    st.write(df_to_use.describe(include='all'))

    # Time series visualizations section
    st.header("Time Series Visualizations")

    # Select a good for time series plot
    good_for_plot = st
