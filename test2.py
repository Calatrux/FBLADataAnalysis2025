import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from datetime import datetime, timedelta
import seaborn as sns

# Load dataset
df = pd.read_csv('Air_Quality.csv')  # Replace with your actual dataset
df.set_index('Start_Date', inplace=True)
# Convert date column to datetime format
df.index = pd.to_datetime(df.index)

df.sort_index(inplace=True)

# Filter for AQI-related rows (modify as needed)
df = df[df['Name'].str.contains('PM 2.5', case=False, na=False)]

# test/train split
train = df.loc[df.index < '2014-01-01']
test = df.loc[df.index >= '2014-01-01']

def create_features(df):
    """
    Create time series features based on time series index.
    """
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df

df = create_features(df)

train = create_features(train)
test = create_features(test)

# Use additional features
FEATURES = ['year', 'month', 'dayofweek', 'hour']
TARGET = 'Data Value'

X_train = train[FEATURES]
y_train = train[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]

reg = xgb.XGBRegressor(
    booster='gbtree',
    n_estimators=500,
    early_stopping_rounds=50,
    objective='reg:squarederror',
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8
)

reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=100)

# fig, ax = plt.subplots(figsize=(10, 8))
# sns.boxplot(data=df, x='year', y='Data Value', ax=ax)
# ax.set_title('PM2.5 Levels by Periods of Time')
# plt.xticks(rotation=90)
# plt.show()

test['prediction'] = reg.predict(X_test)
print(test['prediction'])
print(X_test)
df = df.merge(test[['prediction']], how='left', left_index=True, right_index=True)
ax = df[['Data Value']].plot(figsize=(15, 5))
df['prediction'].plot(ax=ax, style='.')
plt.legend(['Truth Data', 'Predictions'])
ax.set_title('Raw Data and Prediction')
plt.show()