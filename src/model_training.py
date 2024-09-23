import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define paths to data files
train_data_path = 'C:\\Users\\hayyu.ragea\\AppData\\Local\\Programs\\Python\\Python312\\Rossmann_Sales_Forecasting_Project\\data\\train.csv'
test_data_path = 'C:\\Users\\hayyu.ragea\\AppData\\Local\\Programs\\Python\\Python312\\Rossmann_Sales_Forecasting_Project\\data\\test.csv'

# Load data
def load_data(train_path, test_path):
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        return train_df, test_df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

# Preprocess data
def preprocess_data(train_df, test_df):
    # Example feature engineering; adjust based on your actual data
    train_df['Date'] = pd.to_datetime(train_df['Date'])
    test_df['Date'] = pd.to_datetime(test_df['Date'])
    
    # Drop or fill missing values as needed
    train_df.fillna(0, inplace=True)
    test_df.fillna(0, inplace=True)
    
    # Feature selection
    features = ['Store', 'DayOfWeek', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday',
                'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear',
                'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear']
    
    X_train = train_df[features]
    y_train = train_df['Sales']  # Assuming 'Sales' is the target variable
    
    X_test = test_df[features]
    
    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, y_train, X_test_scaled

# Create LSTM model
def create_lstm_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    return model

# Main function
def main():
    try:
        # Load and preprocess data
        train_df, test_df = load_data(train_data_path, test_data_path)
        X_train, y_train, X_test = preprocess_data(train_df, test_df)
        
        # Reshape for LSTM
        time_step = 10  # Define your time steps based on data and model requirements
        X_train_reshaped = np.reshape(X_train, (X_train.shape[0], time_step, -1))
        X_test_reshaped = np.reshape(X_test, (X_test.shape[0], time_step, -1))
        
        # Create and train model
        model = create_lstm_model(input_shape=(time_step, X_train.shape[1]))
        history = model.fit(X_train_reshaped, y_train, epochs=10, batch_size=64, validation_split=0.2, verbose=1)
        
        # Save the model
        model.save('lstm_model.h5')
        
        logging.info("Model training complete and saved as 'lstm_model.h5'")
    
    except Exception as e:
        logging.error(f"Error in model training: {e}")

if __name__ == "__main__":
    main()
