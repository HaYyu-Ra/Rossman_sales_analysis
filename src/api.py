from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
from typing import List
from datetime import datetime
import uvicorn

# Define the FastAPI app
app = FastAPI()

# Define the input data model for the API (for data validation)
class SalesData(BaseModel):
    Store: int
    DayOfWeek: int
    Date: str
    Open: int
    Promo: int
    StateHoliday: str
    SchoolHoliday: int
    StoreType: str
    Assortment: str
    CompetitionDistance: float
    CompetitionOpenSinceMonth: int
    CompetitionOpenSinceYear: int
    Promo2: int
    Promo2SinceWeek: int
    Promo2SinceYear: int
    PromoInterval: str

# Load the trained machine learning model
model_path = "models/saved_models/sales_forecasting_model.pkl"

def load_model(model_path: str):
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model

# Load the serialized model
try:
    model = load_model(model_path)
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Error loading model: {e}")

# Function to preprocess input data
def preprocess_input(data: SalesData):
    # Convert date string to datetime object
    data.Date = pd.to_datetime(data.Date)

    # Example of more preprocessing steps:
    # Convert categorical features to numeric (you can add more preprocessing based on your feature engineering)
    data.StateHoliday = 0 if data.StateHoliday == '0' else 1
    data.StoreType = 1 if data.StoreType == 'a' else (2 if data.StoreType == 'b' else (3 if data.StoreType == 'c' else 4))
    data.Assortment = 1 if data.Assortment == 'a' else (2 if data.Assortment == 'b' else 3)
    data.PromoInterval = 1 if data.PromoInterval in ['Feb', 'May', 'Aug', 'Nov'] else 0

    # Return the processed input as a numpy array
    processed_input = np.array([
        data.Store, data.DayOfWeek, data.Open, data.Promo,
        data.StateHoliday, data.SchoolHoliday, data.StoreType,
        data.Assortment, data.CompetitionDistance, data.CompetitionOpenSinceMonth,
        data.CompetitionOpenSinceYear, data.Promo2, data.Promo2SinceWeek,
        data.Promo2SinceYear, data.PromoInterval
    ]).reshape(1, -1)

    return processed_input

# Root endpoint to check if API is working
@app.get("/")
def read_root():
    return {"message": "Welcome to the Rossmann Sales Forecasting API"}

# Prediction endpoint
@app.post("/predict/")
def predict_sales(sales_data: SalesData):
    try:
        # Preprocess input data
        input_data = preprocess_input(sales_data)

        # Make prediction using the loaded model
        prediction = model.predict(input_data)
        prediction_value = float(prediction[0])

        # Return the prediction result
        return {
            "prediction": prediction_value,
            "message": "Sales prediction successful"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {e}")

# For testing locally
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
