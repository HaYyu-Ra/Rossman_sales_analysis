import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """
    Load data from a CSV file.
    :param file_path: Path to the CSV file.
    :return: pandas DataFrame with the data.
    """
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    """
    Preprocess the data for LSTM model.
    :param data: pandas DataFrame with the data.
    :return: Scaled and reshaped data suitable for LSTM input.
    """
    # Example: Assume the 'Sales' column is the target variable
    data = data[['Sales']]  # Select the relevant column
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Create dataset with time steps
    def create_dataset(data, time_step=1):
        X, y = [], []
        for i in range(len(data) - time_step - 1):
            X.append(data[i:(i + time_step), 0])
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)
    
    time_step = 10  # Number of time steps for LSTM
    X, y = create_dataset(scaled_data, time_step)
    
    # Reshape for LSTM [samples, time steps, features]
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    return X, y, scaler

def build_lstm_model(input_shape):
    """
    Build and compile the LSTM model.
    :param input_shape: Shape of the input data [time steps, features].
    :return: Compiled LSTM model.
    """
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def main():
    # File paths
    data_file_path = 'C:\\Users\\hayyu.ragea\\AppData\\Local\\Programs\\Python\\Python312\\Rossmann_Sales_Forecasting_Project\\data\\train.csv'

    # Load and preprocess data
    data = load_data(data_file_path)
    X, y, scaler = preprocess_data(data)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Build and train the model
    model = build_lstm_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=1)

    # Evaluate the model
    test_loss = model.evaluate(X_test, y_test, verbose=1)
    print(f'Test Loss: {test_loss}')

    # Save the model
    model.save('lstm_model.h5')

if __name__ == "__main__":
    main()
