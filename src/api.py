import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Load the trained models and scaler
rf_model_path = r"C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\Rossmann_Sales_Forecasting_Project\models\saved_models\random_forest_model_28-09-2024-14-14-04.pkl"
lstm_model_path = r"C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\Rossmann_Sales_Forecasting_Project\models\saved_models\lstm_model_28-09-2024-14-14-04.h5"
scaler_path = r"C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\Rossmann_Sales_Forecasting_Project\models\saved_models\minmax_scaler_28-09-2024-14-14-04.pkl"

# Load the Random Forest model and scaler
rf_model = joblib.load(rf_model_path)
scaler = joblib.load(scaler_path)

# Load the LSTM model using TensorFlow/Keras
lstm_model = tf.keras.models.load_model(lstm_model_path)

# Initialize the MinMaxScaler for LSTM scaling (ensure it’s defined here)
scaler_lstm = joblib.load(scaler_path)  # Adjust this line if you're loading a specific LSTM scaler

# Initialize FastAPI
app = FastAPI()

# Define request body for predictions
class SalesPredictionRequest(BaseModel):
    Store: int
    Day: int
    Month: int
    Year: int
    Weekday: int
    Weekend: int
    IsMonthStart: int
    IsMonthEnd: int
    DaysToHoliday: int
    StateHoliday: str  # Add StateHoliday to the request for flexibility

# Define a prediction endpoint for Random Forest
@app.post("/predict_rf/")
async def predict_rf(request: SalesPredictionRequest):
    try:
        # Prepare input data as DataFrame
        input_data = pd.DataFrame([request.dict()])
        input_data.fillna(0, inplace=True)  # Fill NaNs as in preprocessing
        prediction = rf_model.predict(input_data)
        return {"predicted_sales": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Define a prediction endpoint for LSTM
@app.post("/predict_lstm/")
async def predict_lstm(request: SalesPredictionRequest):
    try:
        # Prepare input data for LSTM prediction
        input_data = pd.DataFrame([request.dict()])

        # Define the expected columns for prediction
        expected_columns = [
            "Store", "Day", "Month", "Year", "Weekday", 
            "Weekend", "IsMonthStart", "IsMonthEnd", 
            "DaysToHoliday", "StateHoliday"
        ]

        # Check for missing features and ensure the input has all expected features
        for col in expected_columns:
            if col not in input_data.columns:
                raise HTTPException(status_code=400, detail=f"Missing feature: {col}")

        # Reorder columns to match the scaler's expectations
        input_data = input_data[expected_columns]

        # Fill any NaN values in the input data
        input_data.fillna(0, inplace=True)  # Fill NaNs as needed

        # Scale the input data using the fitted scaler
        input_data_scaled = scaler_lstm.transform(input_data)  # Transform input (1, 10)

        # Reshape for LSTM: (samples, time steps, features)
        input_data_lstm = input_data_scaled.reshape((1, 1, input_data_scaled.shape[1]))

        # Make prediction using the LSTM model
        prediction = lstm_model.predict(input_data_lstm)

        return {"predicted_sales": prediction[0][0]}
    
    except ValueError as ve:
        # Handle specific value errors from the scaler or model
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        # Handle general exceptions
        raise HTTPException(status_code=400, detail=str(e))

# Run the API using Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
