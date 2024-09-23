from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import json
import logging

# Initialize FastAPI app
app = FastAPI()

# Load the trained LSTM model
model_path = 'lstm_model.h5'
model = tf.keras.models.load_model(model_path)

# Initialize MinMaxScaler used during training
scaler = MinMaxScaler(feature_range=(0, 1))

# Define a request model
class PredictionRequest(BaseModel):
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

# Helper function to preprocess input data
def preprocess_input(data: PredictionRequest) -> np.ndarray:
    # Convert input data to DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Example: Preprocessing should be consistent with training preprocessing
    # This is a placeholder; actual preprocessing will depend on your model and feature engineering
    input_df = input_df[['Store', 'DayOfWeek', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday', 
                         'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear',
                         'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear']]
    
    # Assume the scaler was fit on similar data during training
    scaled_data = scaler.transform(input_df)

    # Create dataset with time steps (this example uses a fixed time_step of 10)
    time_step = 10
    def create_dataset(data, time_step=1):
        X = []
        for i in range(len(data) - time_step):
            X.append(data[i:(i + time_step), 0])
        return np.array(X)
    
    X = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    return X

# Define the prediction endpoint
@app.post("/predict/")
async def predict(request: PredictionRequest):
    try:
        # Preprocess the input data
        X = preprocess_input(request)

        # Make predictions
        predictions = model.predict(X)
        
        # Return the predictions
        return {"prediction": predictions.tolist()}

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# Run the app (for development purposes; use a proper ASGI server in production)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
