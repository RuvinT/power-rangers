#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 11:56:40 2023

@author: ruvinjagoda
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM
import pickle
from sklearn.preprocessing import MinMaxScaler


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM
import pickle
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
df = pd.read_csv('ieso_final_dataset.csv')
# ... Perform the steps to convert columns and filter the data as shown in your previous code ...
# Convert the 'Datetime' column to a datetime object
df['Datetime'] = pd.to_datetime(df['Datetime'], format="%Y-%m-%d %H:%M:%S", dayfirst=True)

# Filter the rows based on the 'Datetime' column
df = df[df['Datetime'].dt.year >= 2023]
# Normalize the input features and target variable
scaler = MinMaxScaler()
# Specify the fields/columns that need to be converted
fields_to_convert = ['Market Demand (MW)', 'HOEP ($/MWh)', 'Nuclear Supply (MW)', 'Gas Supply (MW)',
                     'Hydro Supply (MW)', 'Wind Supply (MW)', 'Solar Supply (MW)', 'Biofuel Supply (MW)',
                     'Total Supply (MW)']
df[fields_to_convert] = scaler.fit_transform(df[fields_to_convert])

features = df.copy()






# Load the model from the file
with open('model_ts.pkl', 'rb') as file:
    model = pickle.load(file)


# Get data from 2023 April 16 onwards and up to 336 data points in the future
start_date = pd.to_datetime('2023-04-16')
future_data_points = 336

prediction_data = features[features['Datetime'] >= start_date].iloc[-future_data_points:].copy()
prediction_data['Datetime'] = scaler.fit_transform(prediction_data['Datetime'].dt.hour.values.reshape(-1, 1))
prediction_data['Ontario Demand (MW)'] = scaler.fit_transform(prediction_data['Ontario Demand (MW)'].values.reshape(-1, 1))


print("shape of data",prediction_data.shape)

# Reshape the data to have one additional dimension with size 1
prediction_data_reshaped = prediction_data.values.reshape(1, prediction_data.shape[0], prediction_data.shape[1])

print("Shape of data:", prediction_data_reshaped.shape)

prediction = model.predict(prediction_data_reshaped)

# Inverse transform the scaled predictions
original_predictions = scaler.inverse_transform(prediction)
original_predictions = original_predictions.flatten()

prediction_length = future_data_points
# list called test_dates




from datetime import timedelta

# Create an empty array to store the future time points
future_times = []

# Generate the future time points
for i in range(prediction_length):
    future_time = start_date + timedelta(hours=i)
    future_times.append(future_time)

import matplotlib.pyplot as plt

true_val = scaler.inverse_transform(prediction_data["Ontario Demand (MW)"].values.reshape(-1, 1))
pred_val = original_predictions
# Plot the test data and predicted data
plt.plot(future_times,true_val , label='Actual Data')
plt.plot(future_times,pred_val , label='Predicted Data')

# Set the x-axis label
plt.xlabel('Date')

# Set the y-axis label
plt.ylabel('Ontario Demand (MW)')

# Add a title to the graph
plt.title('Actual Data vs Predicted Data')

# Add a legend
plt.legend()

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45)

# Display the graph
plt.show()



import requests
import json


url = "https://s6eegsh5h6.execute-api.us-east-2.amazonaws.com/EnergyPredit-Insert"

# Create an empty list to store the data objects
data_list = []


# Iterate over the future times and pred_val
for date, value in zip(future_times, pred_val):
    data_obj = {
        "Date": date.strftime("%Y-%m-%d"),  # Convert datetime object to string format
        "Hour": str(date.hour),  # Convert hour to string format
        "EnergyType": "Total Supply (MW)",
        "EnergyPrice": str(value)  # Convert energy price to string format
    }
    data_list.append(data_obj)

# Create the payload with the list of data objects
'''
for obj in data_list :
    
    payload = json.dumps(obj)
    print(payload)
    headers = {
        'Content-Type': 'application/json'
    }
    
    response = requests.request("POST", url, headers=headers, data=payload)
    
    print(response.text)
'''
