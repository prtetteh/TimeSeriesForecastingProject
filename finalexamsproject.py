import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from statsmodels.tsa.holtwinters import SimpleExpSmoothing

# Path to the directory containing your CSV files
data_path = '/content/drive/MyDrive/TimesSeries/FinalExams/'

# List of CSV file names
csv_files = ['Beans (dry).csv', 'Cassava.csv', 'Chili (red).csv', 'Maize.csv',
             'Oranges (big size).csv', 'Peas (fresh).csv', 'Potatoes (Irish).csv',
             'Sorghum.csv', 'Tomatoes.csv']

# Dictionary to store DataFrames
dataframes = {}


def parse_date(df):
    """Parses the date column into a DateTimeIndex."""
    df['date'] = pd.to_datetime(df['mp_year'].astype(str) + '-' + df['mp_month'].astype(str) + '-01')
    return df


def load_and_process_data():
    """Loads CSV files, parses dates, and handles missing values."""
    global dataframes

    for file in csv_files:
        file_path = os.path.join(data_path, file)

        try:
            df = pd.read_csv(file_path)

            if 'mp_year' in df.columns and 'mp_month' in df.columns:
                df = parse_date(df)
                df.set_index('date', inplace=True)
            else:
                st.error(f"Error: 'mp_year' or 'mp_month' columns not found in {file}")
                continue

            dataframes[file] = df.fillna(method='ffill')  # Forward fill missing values

        except FileNotFoundError:
            st.error(f"Error: File {file} not found")
            continue


def display_data_overview():
    """Displays data overview for each DataFrame."""
    for name, df in dataframes.items():
        st.header(f"Data Overview: {name}")
        st.write(df.head())
        st.write(df.info())


def explore_time_series():
    """Visualizes time series data for each good."""
    for name, df in dataframes.items():
        st.header(f"Time Series: {name.split('.')[0]}")
        price_column = 'mp_price' if 'mp_price' in df.columns else 'price'

        if price_column:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df.index, df[price_column], label=name.split('.')[0])
            ax.set_title(f"Time Series Plot of {name.split('.')[0]} Prices")
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.xticks(rotation=45)
            ax.legend()
            st.pyplot(fig)
        else:
            st.warning(f"No price column found in {name}")


def analyze_correlations():
    """Calculates and displays correlation matrix and heatmap."""
    price_data = []
    for name, df in dataframes.items():
        price_column = 'mp_price' if 'mp_price' in df.columns else 'price'
        if price_column:
            price_data.append(df[price_column].reset_index(drop=True))

    if price_data:
        price_df = pd.DataFrame(price_data).transpose()
        price_df.columns = list(dataframes.keys())
