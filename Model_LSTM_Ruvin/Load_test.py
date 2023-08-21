#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 01:06:31 2023

@author: ruvinjagoda
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM
import pickle
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import timedelta
import requests
import json

# Load the dataset
df = pd.read_csv('ieso_final_dataset.csv')

# Convert the 'Datetime' column to a datetime object
df['Datetime'] = pd.to_datetime(df['Datetime'], format="%Y-%m-%d %H:%M:%S", dayfirst=True)

# Filter the rows based on the 'Datetime' column
df = df[df['Datetime'].dt.year >= 2023]

# Normalize the input features and target variable
scaler = MinMaxScaler()
fields_to_convert = ['Market Demand (MW)', 'HOEP ($/MWh)', 'Nuclear Supply (MW)', 'Gas Supply (MW)',
                     'Hydro Supply (MW)', 'Wind Supply (MW)', 'Solar Supply (MW)', 'Biofuel Supply (MW)',
                     'Total Supply (MW)']
df[fields_to_convert] = scaler.fit_transform(df[fields_to_convert])

features = df.copy()

# Load the model from the file
with open('model_Ontario Demand (MW).pkl', 'rb') as file:
    model = pickle.load(file)

# Get data from 2023 April 16 onwards and up to 336 data points in the future
start_date = pd.to_datetime('2023-04-16')
future_data_points = 336

prediction_data = features[features['Datetime'] <= start_date].iloc[-future_data_points:].copy()
prediction_data['Datetime'] = scaler.fit_transform(prediction_data['Datetime'].dt.hour.values.reshape(-1, 1))
prediction_data['Ontario Demand (MW)'] = scaler.fit_transform(prediction_data['Ontario Demand (MW)'].values.reshape(-1, 1))

# Reshape the data to have one additional dimension with size 1
prediction_data_reshaped = prediction_data.values.reshape(1, prediction_data.shape[0], prediction_data.shape[1])

# Add random values between -5 and 5 to prediction_data_reshaped for stress testing
random_values = np.random.uniform(low=-50, high=5, size=(prediction_data_reshaped.shape[1], prediction_data_reshaped.shape[2]))
#prediction_data_reshaped[0, :, :] += random_values

prediction = model.predict(prediction_data_reshaped)

# Inverse transform the scaled predictions
original_predictions = scaler.inverse_transform(prediction)
original_predictions = original_predictions.flatten()

prediction_length = future_data_points

# Create an empty array to store the future time points
future_times = []

# Generate the future time points
for i in range(prediction_length):
    future_time = start_date + timedelta(hours=i)
    future_times.append(future_time)

# Plot the test data and predicted data
true_val = features[features['Datetime'] >= start_date].iloc[:future_data_points].copy()
true_val = true_val['Ontario Demand (MW)'].values.reshape(-1, 1)
pred_val = original_predictions

plt.plot(future_times, true_val, label='Actual Data')
plt.plot(future_times, pred_val, label='Predicted Data')

plt.xlabel('Date')
plt.ylabel('Ontario Demand (MW)')
plt.title('Actual Data vs Predicted Data')
plt.legend()
plt.xticks(rotation=45)
plt.show()

# API Call
url = "https://s6eegsh5h6.execute-api.us-east-2.amazonaws.com/EnergyPredit-Insert"
data_list = []

# Iterate over the future times and pred_val
for date, value in zip(future_times, pred_val):
    data_obj = {
        "Date": date.strftime("%Y-%m-%d"),  # Convert datetime object to string format
        "Hour": str(date.hour),  # Convert hour to string format
        "EnergyType": "Ontario_Demand",
        "EnergyPrice": str(value)  # Convert energy price to string format
    }
    data_list.append(data_obj)
'''
# Send API requests
for obj in data_list:
    payload = json.dumps(obj)
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    print(response.text)
'''