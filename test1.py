import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from datetime import datetime, timedelta

# Load dataset
data = pd.read_csv('Air_Quality.csv')  # Replace with your actual dataset

data = data.head(100)
print(data.head())

# Convert date column to datetime format
data['Start_Date'] = pd.to_datetime(data['Start_Date'])
data = data.sort_values(by='Start_Date')

# Filter for AQI-related rows (modify as needed)
data = data[data['Name'].str.contains('AQI|PM2.5', case=False, na=False)]

# Resample data to ensure consistent time steps (fill missing dates with NaN)
data = data.set_index('Start_Date').resample('D').asfreq()

# Interpolate missing AQI values
data['Data Value'] = data['Data Value'].interpolate(method='linear')

# Reset index
data = data.reset_index().rename(columns={'index': 'Start_Date'})

# Feature engineering (example features, modify based on dataset)
data['Year'] = data['Start_Date'].dt.year
data['Month'] = data['Start_Date'].dt.month
data['Day'] = data['Start_Date'].dt.day

data['Lag_1'] = data['Data Value'].shift(1)  # Previous day's AQI
data['Lag_7'] = data['Data Value'].shift(7)  # AQI from 7 days ago
data.dropna(inplace=True)


print(data.head())