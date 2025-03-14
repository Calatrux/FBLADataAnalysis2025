import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from datetime import datetime, timedelta
import seaborn as sns
import re
from xgboost import XGBRegressor

stored_predictions = []
EMISSIONS_REDUCTION_FACTOR = 0.8

# Load dataset
df = pd.read_csv('NYCAQ2.csv')  # Replace with your actual dataset

def parse_year_from_timeperiod(tp):
    match = re.search(r'\d{4}', tp)
    return int(match.group(0)) if match else 2000

def parse_time_period(tp):
    text = tp.lower()
    if 'summer' in text:
        period_type = 'Summer'
    elif 'winter' in text:
        period_type = 'Winter'
    else:
        period_type = 'Annual'
    match = re.search(r'\d{4}', text)
    year = int(match.group(0)) if match else 2000
    return period_type, year

# Create a Year column from the TimePeriod
df['Year'] = df['TimePeriod'].apply(parse_year_from_timeperiod)

df[['PeriodType', 'PeriodYear']] = df['TimePeriod'].apply(
    lambda x: parse_time_period(x)
).tolist()

df_grouped = df.groupby(['PeriodType','PeriodYear'])['Mean mcg/m3'].mean().reset_index()

# Move the AI predictions before the visualization
prediction_results = []
for ptype in df_grouped['PeriodType'].unique():
    sub = df_grouped[df_grouped['PeriodType'] == ptype].copy()
    X = sub[['PeriodYear']]
    y = sub['Mean mcg/m3']
    model = XGBRegressor(
        random_state=42,
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8
    )
    model.fit(X, y)

    last_year = sub['PeriodYear'].max()
    current_data = sub.copy()

    # Predict each future period in sequence, then re-train on the new data
    for i in range(1, 27):
        next_year = last_year + i
        future_df = pd.DataFrame({'PeriodYear': [next_year]})
        val = model.predict(future_df)[0]


        current_data.loc[len(current_data)] = [ptype, next_year, val]
        model.fit(current_data[['PeriodYear']], current_data['Mean mcg/m3'])

        prediction_results.append({
            'PeriodType': ptype,
            'PeriodYear': next_year,
            'PredictedMean': val
        })

        if i == 26 and ptype == 'Annual':
            stored_predictions.append(val)
        elif i == 26 and ptype == 'Winter':
            stored_predictions.append(val)
        elif i == 26 and ptype == 'Summer':
            stored_predictions.append(val)

pred_df = pd.DataFrame(prediction_results)

plt.figure(figsize=(10,5))

# Combine historical and future data for each PeriodType
for ptype in df_grouped['PeriodType'].unique():
    historical = df_grouped[df_grouped['PeriodType'] == ptype].copy()
    future = pred_df[pred_df['PeriodType'] == ptype].copy()
    future = future.sort_values('PeriodYear')

    # Rename columns for a uniform plot
    historical = historical.rename(columns={'Mean mcg/m3': 'Value'})
    future = future.rename(columns={'PredictedMean': 'Value'})

    # Combine and sort for continuous plotting
    combined = pd.concat([historical[['PeriodYear', 'Value']], future[['PeriodYear', 'Value']]]).sort_values('PeriodYear')

    # After combining and sorting:
    combined = combined[combined['PeriodYear'] > 2020]

    plt.plot(combined['PeriodYear'], combined['Value'], marker='o', linestyle='-', label=ptype)

plt.title('Mean Pollutants Over Time by Period (Historical + Future)')
plt.xlabel('Year')
plt.ylabel('Mean mcg/m3')
plt.legend()
plt.ylim(0, 10)
plt.show()

# Print only the final stored predictions (Annual, Winter, Summer)
print("Annual:", stored_predictions[0])
print("Winter:", stored_predictions[1])
print("Summer:", stored_predictions[2])






