import requests
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the URL of the FastAPI endpoint
url = "http://127.0.0.1:8000/predict/"

# Define the payload for the POST request
payload = {
    "Store": 1,
    "DayOfWeek": 5,
    "Date": "2024-09-19",
    "Open": 1,
    "Promo": 1,
    "StateHoliday": "0",
    "SchoolHoliday": 1,
    "StoreType": "a",
    "Assortment": "c",
    "CompetitionDistance": 500.0,
    "CompetitionOpenSinceMonth": 9,
    "CompetitionOpenSinceYear": 2010,
    "Promo2": 1,
    "Promo2SinceWeek": 10,
    "Promo2SinceYear": 2020,
    "PromoInterval": "Feb"
}

def test_api():
    try:
        # Send POST request
        response = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(payload))
        
        # Check if the request was successful
        if response.status_code == 200:
            logging.info("Request successful.")
            logging.info(f"Response: {response.json()}")
        else:
            logging.error(f"Request failed with status code: {response.status_code}")
            logging.error(f"Response: {response.text}")
    
    except Exception as e:
        logging.error(f"Error during API request: {e}")

if __name__ == "__main__":
    test_api()
