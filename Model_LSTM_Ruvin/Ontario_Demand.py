#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 22:58:43 2023

@author: ruvinjagoda
"""

'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import math
from keras.layers import Dropout
import seaborn as sb
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


# Load the dataset
df = pd.read_csv('ieso_final_dataset.csv')
# ... Perform the steps to convert columns and filter the data as shown in your previous code ...
# Convert the 'Date' column to a datetime object
df['Datetime'] = pd.to_datetime(df['Datetime'], format="%Y-%m-%d %H:%M:%S", dayfirst=True)

# Filter the rows based on the 'Date' column
df = df[df['Datetime'].dt.year >= 2015]
# Normalize the input features and target variable
scaler = MinMaxScaler()
# Specify the fields/columns that need to be converted
fields_to_convert = ['Market Demand (MW)', 'HOEP ($/MWh)', 'Nuclear Supply (MW)', 'Gas Supply (MW)',
                     'Hydro Supply (MW)', 'Wind Supply (MW)', 'Solar Supply (MW)', 'Biofuel Supply (MW)',
                     'Total Supply (MW)']
df[fields_to_convert] = scaler.fit_transform(df[fields_to_convert])
df['Ontario Demand (MW)'] = scaler.fit_transform(df['Ontario Demand (MW)'].values.reshape(-1, 1))

# Split the data into training and testing sets
train_size = int(0.8 * len(df))
train_data = df.iloc[:train_size]
test_data = df.iloc[train_size:]

# Create input sequences
sequence_length = 24  # Number of previous hours to use as input
num_features = len(fields_to_convert)
train_generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(
    train_data[fields_to_convert].values, train_data['Ontario Demand (MW)'].values,
    length=sequence_length, batch_size=32
)

# Build the LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(sequence_length, num_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(train_generator, epochs=20)


future_data = test_data[fields_to_convert].values
future_generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(
    future_data, np.zeros(len(future_data)), length=sequence_length, batch_size=1
)


predictions = model.predict(future_generator)


predictions = scaler.inverse_transform(predictions)


print(predictions)


test_data = scaler.inverse_transform(test_data['Ontario Demand (MW)'].values.reshape(-1, 1))


predictions = predictions.flatten()

prediction_length = 720
# list called test_dates
test_dates = df.iloc[-prediction_length:]['Datetime'].values


pred_dates = test_dates

# Plot the test data and predicted data
plt.plot(test_dates, test_data[-prediction_length:], label='Actual Data')
plt.plot(pred_dates, predictions[-prediction_length:], label='Predicted Data')

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

'''

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
features['Datetime'] = scaler.fit_transform(features['Datetime'].dt.hour.values.reshape(-1, 1))
features['Ontario Demand (MW)'] = scaler.fit_transform(df['Ontario Demand (MW)'].values.reshape(-1, 1))

x_features = []
y_target = []

for i in range(336, (len(df)-336)):
    x_features.append(features[i-336:i])
    y_target.append(features['Ontario Demand (MW)'][i:i+336])
    
    
# Convert the x_train and y_train to numpy arrays 
x_features, y_target = np.array(x_features), np.array(y_target)


features = x_features
target = y_target

print(features.shape,target.shape)
 
X_train, X_valid, Y_train, Y_valid = train_test_split(
    features, target, test_size=0.2, shuffle=False)
    
# define model architecture

model = Sequential()
model.add(LSTM(500, input_shape=(336,11), return_sequences=True))
model.add(LSTM(400))
model.add(Dense(336))  # Output layer with 120 units

# compile model
model.compile(optimizer='adam', loss='mse')

# train model
history = model.fit(X_train, Y_train, epochs=20, batch_size=32, validation_data=(X_valid, Y_valid))


# Save the model to a file
filename = 'model_Ontario Demand (MW).pkl'
with open(filename, 'wb') as file:
    pickle.dump(model, file)
    
     
# Make a prediction

prediction = model.predict(X_valid[0:1, :, :])

# Inverse transform the scaled predictions
original_predictions = scaler.inverse_transform(prediction)
original_predictions = original_predictions.flatten()
prediction_length = 336
# list called test_dates
start_time = df.iloc[-1:]['Datetime']

from datetime import timedelta

# Create an empty array to store the future time points
future_times = []

# Generate the future time points
for i in range(prediction_length):
    future_time = start_time + timedelta(hours=i)
    future_times.append(future_time)

import matplotlib.pyplot as plt

true_val = scaler.inverse_transform(Y_valid[0:1,:].reshape(-1, 1))
true_val = true_val.flatten()
pred_val = original_predictions

print(true_val)
print(pred_val)
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



'''
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
        "Date": str(date),
        "EnergyType": "Ontario Demand (MW)",
        "EnergyPrice": value
    }
    data_list.append(data_obj)

# Create the payload with the list of data objects
payload = json.dumps(data_list)

headers = {
    'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)

'''







