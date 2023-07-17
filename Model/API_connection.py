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
                     'Total Supply (MW)','Ontario Demand (MW)']
df[fields_to_convert] = scaler.fit_transform(df[fields_to_convert])

features = df.copy()
features['Datetime'] = scaler.fit_transform(features['Datetime'].dt.hour.values.reshape(-1, 1))


x_features = []
y_target = []

for i in range(120, (len(df)-120)):
    x_features.append(features[i-120:i])
    y_target.append(df['Ontario Demand (MW)'][i:i+120])
    
    
# Convert the x_train and y_train to numpy arrays 
x_features, y_target = np.array(x_features), np.array(y_target)


features = x_features
target = y_target

print(features.shape,target.shape)
 
X_train, X_valid, Y_train, Y_valid = train_test_split(
    features, target, test_size=0.2, shuffle=False)



# Load the model from the file
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Make a prediction

prediction = model.predict(X_train[-1:])

# Inverse transform the scaled predictions
original_predictions = scaler.inverse_transform(prediction)

prediction_length = 120
# list called test_dates
start_time = df.iloc[-1:]['Datetime']

from datetime import timedelta

# Create an empty array to store the future time points
future_times = []

# Generate the future time points
for i in range(prediction_length):
    future_time = start_time.iloc[0] + timedelta(hours=i)
    future_times.append(future_time)

import matplotlib.pyplot as plt

true_val = scaler.inverse_transform(Y_valid[-1:])[0][:prediction_length]

pred_val = original_predictions[0][:prediction_length]
# Plot the test data and predicted data
plt.plot(future_times,true_val , label='Actual Data')
plt.plot(future_times,pred_val , label='Predicted Data')

# Set the x-axis label
plt.xlabel('Date')

# Set the y-axis label
plt.ylabel('HOEP ($/MWh)')

# Add a title to the graph
plt.title('Actual Data vs Predicted Data')

# Add a legend
plt.legend()

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45)

# Display the graph
plt.show()

from sklearn.metrics import mean_absolute_percentage_error

# Obtaining predictions from the trained LSTM model
predictions = model.predict(X_valid)

# Calculating the MAPE
mape = mean_absolute_percentage_error(Y_valid, predictions)
print("MAPE:", mape)

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
        "EnergyType": "Ontario_Demand",
        "EnergyPrice": str(value)  # Convert energy price to string format
    }
    data_list.append(data_obj)

# Create the payload with the list of data objects

for obj in data_list :
    
    payload = json.dumps(obj)
    print(payload)
    headers = {
        'Content-Type': 'application/json'
    }
    
    response = requests.request("POST", url, headers=headers, data=payload)
    
    print(response.text)

