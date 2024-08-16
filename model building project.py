# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 15:23:47 2023

@author: nagashree k d
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import seaborn as sns
import scipy
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import probplot
import scipy.stats as stats
import pylab
import pandas as pd

data = pd.read_csv(r"C:\Users\nagashree k d\Documents\rod quantity.csv")


# Now you can work with the 'data' DataFrame

# Check the shape of the DataFrame
print(data.shape)


# Check for null values in the DataFrame
null_values = data.isnull()
null_counts = null_values.sum()
total_null_count = null_counts.sum()

# Print the DataFrame containing True/False values for null presence
print("Null Values DataFrame:")
print(null_values)
print("\nNull Value Counts in Each Column:")
print(null_counts)
print(f"\nTotal Null Values: {total_null_count}")


#Imputation, which involves filling in missing values in a dataset, is not requeard because there is no missing values present


# Load your dataset from the Excel fileimport pandas as pd
from datetime import timedelta
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from math import sqrt

data = pd.read_csv(r"C:\Users\nagashree k d\Documents\rod quantity.csv")

# Convert the "Date Date" column to datetime format
data['Date'] = pd.to_datetime(data['Date'])  # Use the exact column name

# Sort the data by date
data.sort_values(by='Date', inplace=True)

# Calculate the date range for training and testing
test_start_date = data['Date'].max() - timedelta(days=365)
train_data = data[data['Date'] < test_start_date]
test_data = data[data['Date'] >= test_start_date]

# Assuming your data has a "Date" and "quantity" columns
train_data = train_data[['Date', 'Quantity']]  # Include the "quantity" column
test_data = test_data[['Date', 'Quantity']]    # Include the "quantity" column

# Check the shapes of the training and testing DataFrames
print("Training Data Shape:", train_data.shape)
print("Testing Data Shape:", test_data.shape)

print("Training Data:")
print(train_data)

print("Test Data:")
print(test_data)

# Assuming your data has a "Date" and "quantity" columns
X_train = train_data[["Date"]]  # Features for training
y_train = train_data["Quantity"]  # Target variable for training

import pandas as pd

data = pd.read_csv(r"C:\Users\nagashree k d\Documents\rod quantity.csv")

# Calculate the range of the target variable "quantity"
Quantity_range = data['Quantity'].max() - data['Quantity'].min()

# Print the range of the target variable
print("Range of Target Variable 'Quantity':", Quantity_range)





####    Multilinear regression 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load your dataset from the CSV file
data = pd.read_csv(r"C:\Users\nagashree k d\Documents\rod quantity.csv")

# Convert the "Date" column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Sort the data by date
data.sort_values(by='Date', inplace=True)

# Split the data into training and testing sets
test_start_date = data['Date'].max() - pd.DateOffset(years=1)
train_data = data[data['Date'] < test_start_date]
test_data = data[data['Date'] >= test_start_date]
# Assuming your data has a "Date" and "Quantity" columns
X_train = train_data[["Date"]]  # Features for training
y_train = train_data["Quantity"]  # Target variable for training

X_test = test_data[["Date"]]  # Features for testing
y_test = test_data["Quantity"]  # Target variable for testing

# Extract relevant features from the "Date" column
X_train["Year"] = X_train["Date"].dt.year
X_train["Month"] = X_train["Date"].dt.month
X_train["Day"] = X_train["Date"].dt.day

X_test["Year"] = X_test["Date"].dt.year
X_test["Month"] = X_test["Date"].dt.month
X_test["Day"] = X_test["Date"].dt.day

# Drop the original "Date" column
X_train.drop(columns=["Date"], inplace=True)
X_test.drop(columns=["Date"], inplace=True)
# Initialize and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test data
predictions = model.predict(X_test)

# Calculate RMSE and R-squared for evaluation
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("RMSE:", rmse)
print("R-squared:", r2)

# Plot the actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.plot(test_data['Date'], y_test, label='Actual')
plt.plot(test_data['Date'], predictions, label='Predicted', color='red')
plt.xlabel('Date')
plt.ylabel('Quantity')
plt.title('Multiple Linear Regression Forecasting')
plt.legend()
plt.show()





#####    RNN
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from sklearn.metrics import mean_squared_error
from math import sqrt

# Load your time series data
data = pd.read_csv(r"C:\Users\nagashree k d\Documents\rod quantity.csv")

# Normalize data
scaler = MinMaxScaler()
data['Quantity'] = scaler.fit_transform(data['Quantity'].values.reshape(-1, 1))

# Convert data to sequences
sequence_length = 10
sequences = []
for i in range(len(data) - sequence_length + 1):
    sequences.append(data['Quantity'].values[i:i+sequence_length])

sequences = np.array(sequences)

# Split data into training and testing sets
train_size = int(0.8 * len(sequences))
train_data = sequences[:train_size]
test_data = sequences[train_size:]

X_train = train_data[:, :-1]
y_train = train_data[:, -1]
X_test = test_data[:, :-1]
y_test = test_data[:, -1]

# Reshape data to fit RNN input
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build RNN model with SimpleRNN layer
model = Sequential()
model.add(SimpleRNN(50, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Predict on test data
predictions = model.predict(X_test)

# Transform predictions back to original scale
predictions = scaler.inverse_transform(predictions)

# Calculate RMSE for evaluation
rmse = sqrt(mean_squared_error(y_test, predictions))
print("RMSE:", rmse)






####  simple exponential smoothing 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import ParameterGrid

# Load your dataset from the CSV file
data = pd.read_csv(r"C:\Users\nagashree k d\Documents\rod quantity.csv")

# Convert the "Date" column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Sort the data by date
data.sort_values(by='Date', inplace=True)

# Split the data into training and testing sets
test_start_date = data['Date'].max() - pd.DateOffset(years=1)
train_data = data[data['Date'] < test_start_date]
test_data = data[data['Date'] >= test_start_date]

# Assuming your data has a "Date" and "Quantity" columns
y_train = train_data["Quantity"]  # Target variable for training
y_test = test_data["Quantity"]    # Target variable for testing

# Define the parameter grid for tuning
param_grid = {
    'smoothing_level': np.arange(0.1, 1.1, 0.1)
}

# Perform grid search
best_rmse = float('inf')
best_params = {}
for params in ParameterGrid(param_grid):
    smoothing_model = SimpleExpSmoothing(y_train)
    fitted_model = smoothing_model.fit(smoothing_level=params['smoothing_level'])
    predictions = fitted_model.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)
    
    rmse = sqrt(mean_squared_error(y_test, predictions))
    if rmse < best_rmse:
        best_rmse = rmse
        best_params = params

# Fit the best model with the best parameters
best_smoothing_model = SimpleExpSmoothing(y_train)
best_fitted_model = best_smoothing_model.fit(smoothing_level=best_params['smoothing_level'])
best_predictions = best_fitted_model.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)

# Print the best parameters and RMSE
print("Best Parameters:", best_params)
print("Best RMSE:", best_rmse)

# Plot the actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.plot(test_data['Date'], y_test, label='Actual')
plt.plot(test_data['Date'], best_predictions, label='Predicted', color='red')
plt.xlabel('Date')
plt.ylabel('Quantity')
plt.title('Simple Exponential Smoothing with Hyperparameter Tuning')
plt.legend()
plt.show()






## gradient boosting 
import pandas as pd
from datetime import timedelta
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
from datetime import timedelta
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt  # Import matplotlib

# Load your dataset from the Excel file

data = pd.read_csv(r"C:\Users\nagashree k d\Documents\rod quantity.csv")


# Convert the "Date" column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Sort the data by date
data.sort_values(by='Date', inplace=True)

# Calculate the date range for training and testing
test_start_date = data['Date'].max() - timedelta(days=365)
train_data = data[data['Date'] < test_start_date]
test_data = data[data['Date'] >= test_start_date]

# Assuming your data has a "Date" and "Quantity" columns
X_train = train_data[["Date"]]  # Features for training
y_train = train_data["Quantity"]  # Target variable for training

X_test = test_data[["Date"]]  # Features for testing
y_test = test_data["Quantity"]  # Target variable for testing

# Initialize and fit the Gradient Boosting model
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)

# Make predictions on the test data
predictions = gb_model.predict(X_test)

# Calculate RMSE for evaluation
rmse = sqrt(mean_squared_error(y_test, predictions))
print("RMSE:", rmse)

# Plot the actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.plot(test_data['Date'], y_test, label='Actual')
plt.plot(test_data['Date'], predictions, label='Predicted', color='red')
plt.xlabel('Date')
plt.ylabel('Quantity')
plt.title('Gradient Boosting Forecasting')
plt.legend()
plt.show()





### ARIMA [Auto Regression ]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.arima.model import ARIMA

# Load your dataset from the CSV file
data = pd.read_csv(r"C:\Users\nagashree k d\Documents\rod quantity.csv")

# Convert the "Date" column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Sort the data by date
data.sort_values(by='Date', inplace=True)

# Split the data into training and testing sets
test_start_date = data['Date'].max() - pd.DateOffset(years=1)
train_data = data[data['Date'] < test_start_date]
test_data = data[data['Date'] >= test_start_date]

# Assuming your data has a "Date" and "Quantity" columns
y_train = train_data["Quantity"]  # Target variable for training
y_test = test_data["Quantity"]    # Target variable for testing

# Assuming your data has a "Date" and "Quantity" columns
y_train = data.set_index('Date')['Quantity']  # Assuming "Quantity" is your target variable

# Fit ARIMA model to the training data
p, d, q = 1, 1, 1  # ARIMA order parameters
model = ARIMA(y_train, order=(p, d, q))
fitted_model = model.fit()

# Make predictions on the test data
predictions = fitted_model.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1, dynamic=False)

# Calculate RMSE for evaluation
rmse = sqrt(mean_squared_error(y_test, predictions))
print("RMSE:", rmse)

# Plot the actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.plot(test_data['Date'], y_test, label='Actual')
plt.plot(test_data['Date'], predictions, label='Predicted', color='red')
plt.xlabel('Date')
plt.ylabel('Quantity')
plt.title('ARIMA Forecasting')
plt.legend()
plt.show()





#### Naive bayes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt

# Load your dataset from the CSV file
data = pd.read_csv(r"C:\Users\nagashree k d\Documents\rod quantity.csv")

# Convert the "Date" column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Sort the data by date
data.sort_values(by='Date', inplace=True)

# Split the data into training and testing sets
test_start_date = data['Date'].max() - pd.DateOffset(years=1)
train_data = data[data['Date'] < test_start_date]
test_data = data[data['Date'] >= test_start_date]

# Assuming your data has a "Date" and "Quantity" columns
y_train = train_data["Quantity"]  # Target variable for training
y_test = test_data["Quantity"]    # Target variable for testing

# Use the last observed value as the forecast for all future time periods
last_observed_value = y_train.iloc[-1]
forecast = np.full(len(test_data), last_observed_value)

# Calculate RMSE for evaluation
rmse = sqrt(mean_squared_error(y_test, forecast))
print("RMSE:", rmse)

# Plot the actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.plot(test_data['Date'], y_test, label='Actual')
plt.plot(test_data['Date'], forecast, label='Naive Forecast', color='red')
plt.xlabel('Date')
plt.ylabel('Quantity')
plt.title('Naive Forecasting')
plt.legend()
plt.show()


 


##### knn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load your dataset from the CSV file
data = pd.read_csv(r"C:\Users\nagashree k d\Documents\rod quantity.csv")

# Convert the "Date" column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Sort the data by date
data.sort_values(by='Date', inplace=True)

# Split the data into training and testing sets
test_start_date = data['Date'].max() - pd.DateOffset(years=1)
train_data = data[data['Date'] < test_start_date]
test_data = data[data['Date'] >= test_start_date]

# Assuming your data has a "Date" column
train_data["Year"] = train_data["Date"].dt.year
train_data["Month"] = train_data["Date"].dt.month
train_data["Day"] = train_data["Date"].dt.day

test_data["Year"] = test_data["Date"].dt.year
test_data["Month"] = test_data["Date"].dt.month
test_data["Day"] = test_data["Date"].dt.day

# Features and target for training
X_train = train_data[["Year", "Month", "Day"]]
y_train = train_data["Quantity"]

# Features and target for testing
X_test = test_data[["Year", "Month", "Day"]]
y_test = test_data["Quantity"]

# Initialize and fit KNN regressor
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)
knn_predictions = knn.predict(X_test)
knn_rmse = np.sqrt(mean_squared_error(y_test, knn_predictions))

# Initialize and fit Decision Tree regressor
dt = DecisionTreeRegressor(max_depth=5, random_state=0)
dt.fit(X_train, y_train)
dt_predictions = dt.predict(X_test)
dt_rmse = np.sqrt(mean_squared_error(y_test, dt_predictions))

# Plot the actual vs. predicted values for KNN
plt.figure(figsize=(10, 6))
plt.plot(test_data['Date'], y_test, label='Actual')
plt.plot(test_data['Date'], knn_predictions, label='KNN Predicted', color='red')
plt.xlabel('Date')
plt.ylabel('Quantity')
plt.title('KNN Forecasting')
plt.legend()
plt.show()

# Plot the actual vs. predicted values for Decision Tree
plt.figure(figsize=(10, 6))
plt.plot(test_data['Date'], y_test, label='Actual')
plt.plot(test_data['Date'], dt_predictions, label='Decision Tree Predicted', color='green')
plt.xlabel('Date')
plt.ylabel('Quantity')
plt.title('Decision Tree Forecasting')
plt.legend()
plt.show()

print("KNN RMSE:", knn_rmse)
print("Decision Tree RMSE:", dt_rmse)




####linear regression
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt

# Load your dataset and preprocess it

# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Extract year, month, and day as separate columns
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day

# Split the data into features (X) and target (y)
X = data[['Year', 'Month', 'Day']]
y = data['Quantity']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Linear Regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
predictions = model.predict(X_test)

# Calculate RMSE for evaluation
rmse = sqrt(mean_squared_error(y_test, predictions))
print("RMSE:", rmse)




####random forest 
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

# Assuming you have a DataFrame named "data" with columns "Date", "Quantity", and other relevant features

# Convert the "Date" column to a datetime format if it's not already in that format
data['Date'] = pd.to_datetime(data['Date'])

# Sort the data by date
data.sort_values('Date', inplace=True)

# Create lagged features
for lag in range(1, 6):  # Creating lag features for the past 5 time steps
    data[f'lag_{lag}'] = data['Quantity'].shift(lag)

# Drop rows with missing values
data.dropna(inplace=True)

# Splitting data into training and test sets
train_size = int(0.8 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]

# Define features and target variable
X_train = train_data.drop(['Date', 'Quantity'], axis=1)
y_train = train_data['Quantity']
X_test = test_data.drop(['Date', 'Quantity'], axis=1)
y_test = test_data['Quantity']

# Create and train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on the test set
predictions = rf_model.predict(X_test)

# Calculate RMSE (Root Mean Squared Error) for evaluation
rmse = sqrt(mean_squared_error(y_test, predictions))
print("RMSE:", rmse)





####neaural network
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load your dataset from the CSV file
data = pd.read_csv(r"C:\Users\nagashree k d\Documents\rod quantity.csv")

# Convert the "Date" column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Sort the data by date
data.sort_values(by='Date', inplace=True)

# Create a feature "DaysSinceMinDate"
data['DaysSinceMinDate'] = (data['Date'] - data['Date'].min()).dt.days

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

# Normalize the data
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data[['DaysSinceMinDate', 'Quantity']])
test_scaled = scaler.transform(test_data[['DaysSinceMinDate', 'Quantity']])

X_train = train_scaled[:, :-1]
y_train = train_scaled[:, -1]
X_test = test_scaled[:, :-1]
y_test = test_scaled[:, -1]

# Build the neural network model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# Make predictions on the test data
y_pred_scaled = model.predict(X_test)
y_pred = scaler.inverse_transform(np.hstack((X_test, y_pred_scaled)))

# Plot the actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.plot(test_data['Date'], test_data['Quantity'], label='Actual')
plt.plot(test_data['Date'], y_pred[:, -1], label='Predicted', color='red')
plt.xlabel('Date')
plt.ylabel('Quantity')
plt.title('Neural Network Forecasting')
plt.legend()
plt.show()

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(test_data['Quantity'], y_pred[:, -1]))
print("RMSE:", rmse)















