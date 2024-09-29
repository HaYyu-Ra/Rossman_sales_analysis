import os
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.stattools import adfuller, acf, pacf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.losses import MeanAbsoluteError

# Load the Rossmann Store Sales dataset from the specified path
data_path = 'C:/Users/hayyu.ragea/AppData/Local/Programs/Python/Python312/Rossmann_Sales_Forecasting_Project/data/synthetic_sales_data.csv'
data = pd.read_csv(data_path)

# Check column names in the DataFrame
print("Columns in the DataFrame:", data.columns)

# Ensure that the 'Date' column exists (correct case)
if 'Date' in data.columns:
    # Convert the 'Date' column to datetime format
    data['Date'] = pd.to_datetime(data['Date'])
else:
    raise KeyError("The 'Date' column is missing from the dataset.")

# Task 2.1: Feature extraction
data['weekday'] = data['Date'].dt.weekday  # Extract weekday (0=Monday, 6=Sunday)
data['is_weekend'] = (data['weekday'] >= 5).astype(int)  # Mark weekends
data['days_to_holiday'] = (pd.to_datetime('2024-12-25') - data['Date']).dt.days  # Days to next holiday
data['days_after_holiday'] = (data['Date'] - pd.to_datetime('2024-12-25')).dt.days.clip(lower=0)  # Days since last holiday
data['is_beginning_of_month'] = (data['Date'].dt.day <= 7).astype(int)  # Is beginning of the month
data['is_mid_month'] = ((data['Date'].dt.day > 7) & (data['Date'].dt.day <= 14)).astype(int)  # Is mid-month
data['is_end_of_month'] = (data['Date'].dt.day > 14).astype(int)  # Is end of the month

# Handle NaN values by filling them with zeros
data.fillna(0, inplace=True)

# Task 2.1: Prepare features and target variable
X = data.drop(columns=['Sales', 'Date'], errors='ignore')  # Features (exclude target and date)
y = data['Sales']  # Target variable

# Task 2.1: Handle categorical variables using OneHotEncoder
categorical_features = ['Store_Type', 'Store_Status']  # Add categorical features to encode
numerical_features = ['weekday', 'days_to_holiday', 'days_after_holiday']  # Select numerical features to scale

# Create a column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Apply the preprocessing
X = preprocessor.fit_transform(X)

# Task 2.2: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Split dataset

# Task 2.3: Choose a loss function
loss_function = MeanAbsoluteError()  # Define the loss function
print("Loss Function Selected: Mean Absolute Error")

# Task 2.4: Create and fit the Random Forest model using a pipeline
pipeline = Pipeline([
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))  # Initialize RandomForestRegressor
])

pipeline.fit(X_train, y_train)  # Fit the model to the training data

# Task 2.4: Make predictions using the model
y_pred = pipeline.predict(X_test)  # Predict on test set

# Task 2.4: Post Prediction Analysis
# Evaluate model performance using the chosen loss function
mae = loss_function(y_test, y_pred)  # Calculate Mean Absolute Error
print(f'Mean Absolute Error: {mae.numpy()}')  # Output the MAE

# Explore feature importance
importance = pipeline.named_steps['regressor'].feature_importances_  # Get feature importance
features = preprocessor.get_feature_names_out()  # Get feature names after transformation

# Plot feature importance
plt.barh(features, importance)  # Create horizontal bar plot for feature importance
plt.xlabel('Feature Importance')  # Label for x-axis
plt.title('Feature Importance for Sales Prediction')  # Title for the plot
plt.show()  # Show the plot

# Estimate confidence intervals for predictions
confidence_interval = 1.96 * np.std(y_pred) / np.sqrt(len(y_pred))  # 95% confidence interval
print(f'95% Confidence Interval for Predictions: +/- {confidence_interval:.2f}')

# Task 2.5: Serialize the Random Forest model
timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")  # Create timestamp for file name
model_directory = 'C:/Users/hayyu.ragea/AppData/Local/Programs/Python/Python312/Rossmann_Sales_Forecasting_Project/models/saved_models'  # Directory for saving models

# Create the models directory if it does not exist
if not os.path.exists(model_directory):
    os.makedirs(model_directory)  # Create the directory

model_filename = f'{model_directory}/random_forest_model_{timestamp}.pkl'  # File path for saving the model
with open(model_filename, 'wb') as file:
    pickle.dump(pipeline, file)  # Save the model using pickle
print(f'Model saved as: {model_filename}')

# Task 2.5.1: Save the MinMaxScaler model
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(y.values.reshape(-1, 1))  # Fit the scaler on target variable

scaler_filename = f'{model_directory}/minmax_scaler_{timestamp}.pkl'  # File path for saving the scaler model
with open(scaler_filename, 'wb') as file:
    pickle.dump(scaler, file)  # Save the scaler using pickle
print(f'MinMaxScaler saved as: {scaler_filename}')

# Task 2.6: Deep Learning - LSTM Model
data['Sales'] = data['Sales'].astype(float)  # Ensure sales data is float

# Task 2.6.1: Check stationarity with Augmented Dickey-Fuller test
adf_test = adfuller(data['Sales'])
print(f'ADF Statistic: {adf_test[0]}')
print(f'p-value: {adf_test[1]}')

if adf_test[1] > 0.05:
    print("Time series is not stationary. Differencing the data...")
    data['sales_diff'] = data['Sales'].diff().fillna(0)  # Difference the data to make it stationary
else:
    print("Time series is stationary.")
    data['sales_diff'] = data['Sales']  # No differencing needed

# Task 2.6.2: Autocorrelation and Partial Autocorrelation
lag_acf = acf(data['sales_diff'], nlags=20)
lag_pacf = pacf(data['sales_diff'], nlags=20)

# Plot ACF
plt.figure(figsize=(10, 6))
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.title('Autocorrelation Function')

# Plot PACF
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.title('Partial Autocorrelation Function')
plt.show()

# Task 2.6.5: Transform the time series data into supervised learning data
def create_dataset(data, time_step=1):
    X, Y = [], []  # Initialize lists for features and target
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])  # Create feature window
        Y.append(data[i + time_step, 0])  # Create target value
    return np.array(X), np.array(Y)  # Return features and targets as numpy arrays

# Task 2.6.6: Scale the data in (-1, 1) range
scaler_lstm = MinMaxScaler(feature_range=(-1, 1))  # Initialize MinMaxScaler
data_scaled = scaler_lstm.fit_transform(data['sales_diff'].values.reshape(-1, 1))  # Scale the sales data

# Create the dataset for LSTM
time_step = 10
X_lstm, y_lstm = create_dataset(data_scaled, time_step)  # Create datasets for LSTM

# Reshape input to be [samples, time steps, features] which is required for LSTM
X_lstm = X_lstm.reshape(X_lstm.shape[0], X_lstm.shape[1], 1)  # Reshape for LSTM input

# Split into train and test sets for LSTM
X_lstm_train, X_lstm_test, y_lstm_train, y_lstm_test = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)

# Task 2.6.7: Build and compile the LSTM model
model = Sequential()  # Initialize Sequential model
model.add(LSTM(50, return_sequences=True, input_shape=(X_lstm_train.shape[1], 1)))  # First LSTM layer
model.add(LSTM(50, return_sequences=False))  # Second LSTM layer
model.add(Dense(1))  # Output layer

model.compile(loss='mean_squared_error', optimizer='adam')  # Compile the model
print("LSTM model compiled.")

# Task 2.6.8: Train the LSTM model
model.fit(X_lstm_train, y_lstm_train, epochs=50, batch_size=32, verbose=1)  # Train the model

# Task 2.6.9: Evaluate the LSTM model
y_lstm_pred = model.predict(X_lstm_test)  # Make predictions on test set
y_lstm_pred = scaler_lstm.inverse_transform(y_lstm_pred)  # Inverse transform predictions to original scale

# Calculate MAE for LSTM
lstm_mae = mean_absolute_error(y_lstm_test, y_lstm_pred)
print(f'LSTM Mean Absolute Error: {lstm_mae}')

# Task 2.6.10: Serialize the LSTM model
lstm_model_filename = f'{model_directory}/lstm_model_{timestamp}.h5'  # File path for saving LSTM model
model.save(lstm_model_filename)  # Save LSTM model
print(f'LSTM model saved as: {lstm_model_filename}')

# Task 2.6.11: Plot LSTM predictions vs actual
plt.figure(figsize=(14, 7))
plt.plot(y_lstm_test, label='Actual Sales', color='blue')  # Actual values
plt.plot(y_lstm_pred, label='Predicted Sales', color='orange')  # Predicted values
plt.title('LSTM Model Predictions vs Actual Sales')
plt.xlabel('Time Step')
plt.ylabel('Sales')
plt.legend()
plt.show()  # Show the plot

