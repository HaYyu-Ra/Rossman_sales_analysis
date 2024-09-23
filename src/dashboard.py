import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import load_model

# Paths to your models and data
MODEL_PATH = 'path/to/your/saved/model.pkl'
DEEP_MODEL_PATH = 'path/to/your/deep_learning_model.h5'
DATA_PATH = 'C:/Users/hayyu.ragea/AppData/Local/Programs/Python/Python312/Rossmann_Sales_Forecasting_Project/data/test.csv'

# Load your models
@st.cache
def load_model(model_path):
    return joblib.load(model_path)

@st.cache
def load_deep_model(model_path):
    return load_model(model_path)

model = load_model(MODEL_PATH)
deep_model = load_deep_model(DEEP_MODEL_PATH)

# Load and preprocess data
@st.cache
def load_data(data_path):
    df = pd.read_csv(data_path)
    # Preprocess as required
    return df

data = load_data(DATA_PATH)

# Preprocessing function
def preprocess_data(df):
    # Example preprocessing steps
    df['Date'] = pd.to_datetime(df['Date'])
    df['Store'] = df['Store'].astype(str)
    df['DayOfWeek'] = df['Date'].dt.dayofweek + 1
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['DaysToHoliday'] = (df['Date'] - pd.to_datetime(df['StateHoliday'])).dt.days
    df['IsWeekend'] = (df['DayOfWeek'] > 5).astype(int)
    df['IsHoliday'] = (df['StateHoliday'] != '0').astype(int)
    return df

data = preprocess_data(data)

# Sidebar
st.sidebar.header('Input Parameters')

# Input fields for user to enter parameters
store = st.sidebar.selectbox('Store', data['Store'].unique())
day_of_week = st.sidebar.slider('Day of Week', 1, 7, 1)
date = st.sidebar.date_input('Date', pd.to_datetime('today'))
open_store = st.sidebar.selectbox('Open Store', [0, 1])
promo = st.sidebar.selectbox('Promo', [0, 1])
state_holiday = st.sidebar.selectbox('State Holiday', ['0', 'a', 'b', 'c'])
school_holiday = st.sidebar.selectbox('School Holiday', [0, 1])
store_type = st.sidebar.selectbox('Store Type', ['a', 'b', 'c', 'd'])
assortment = st.sidebar.selectbox('Assortment', ['a', 'b', 'c'])
competition_distance = st.sidebar.number_input('Competition Distance', 0, 10000, 500)
competition_open_since_month = st.sidebar.slider('Competition Open Since Month', 1, 12, 1)
competition_open_since_year = st.sidebar.slider('Competition Open Since Year', 2000, 2024, 2010)
promo2 = st.sidebar.selectbox('Promo2', [0, 1])
promo2_since_week = st.sidebar.slider('Promo2 Since Week', 1, 52, 1)
promo2_since_year = st.sidebar.slider('Promo2 Since Year', 2000, 2024, 2020)
promo_interval = st.sidebar.selectbox('Promo Interval', ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

# Prepare input data for prediction
input_data = pd.DataFrame({
    'Store': [store],
    'DayOfWeek': [day_of_week],
    'Date': [date],
    'Open': [open_store],
    'Promo': [promo],
    'StateHoliday': [state_holiday],
    'SchoolHoliday': [school_holiday],
    'StoreType': [store_type],
    'Assortment': [assortment],
    'CompetitionDistance': [competition_distance],
    'CompetitionOpenSinceMonth': [competition_open_since_month],
    'CompetitionOpenSinceYear': [competition_open_since_year],
    'Promo2': [promo2],
    'Promo2SinceWeek': [promo2_since_week],
    'Promo2SinceYear': [promo2_since_year],
    'PromoInterval': [promo_interval]
})

# Predict using the machine learning model
if st.button('Predict with ML Model'):
    prediction_ml = model.predict(input_data)
    st.write('Predicted Sales (ML Model):', prediction_ml[0])

# Predict using the deep learning model
def preprocess_deep_model_input(df):
    # Additional preprocessing for deep learning model
    return df

input_data_deep = preprocess_deep_model_input(input_data)

if st.button('Predict with Deep Learning Model'):
    prediction_dl = deep_model.predict(input_data_deep)
    st.write('Predicted Sales (Deep Learning Model):', prediction_dl[0][0])

# Visualization
st.subheader('Sales Data Visualization')

# Example plot of sales data
st.write('Example plot of sales data.')
fig, ax = plt.subplots()
sns.lineplot(x='Date', y='Sales', data=data)
st.pyplot(fig)

# Additional visualizations
st.subheader('Distribution of Sales and Customers')
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
sns.histplot(data['Sales'], kde=True, ax=ax[0])
ax[0].set_title('Sales Distribution')
sns.histplot(data['Customers'], kde=True, ax=ax[1])
ax[1].set_title('Customers Distribution')
st.pyplot(fig)

st.subheader('Sales vs. Promotions')
fig, ax = plt.subplots()
sns.boxplot(x='Promo', y='Sales', data=data)
ax.set_title('Sales vs. Promotions')
st.pyplot(fig)

st.subheader('Sales by Store Type')
fig, ax = plt.subplots()
sns.boxplot(x='StoreType', y='Sales', data=data)
ax.set_title('Sales by Store Type')
st.pyplot(fig)
