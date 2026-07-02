import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Page Configuration
st.set_page_config(page_title="Taxi Trip Pricing Predictor", layout="centered")

# Assets Load Seiyya (Model, Scaler, Columns)
@st.cache_resource
def load_assets():
    with open('model.pkl', 'rb') as m_file:
        model = pickle.load(m_file)
    with open('scaler.pkl', 'rb') as s_file:
        scaler = pickle.load(s_file)
    return model, scaler

try:
    model, scaler = load_assets()
except FileNotFoundError:
    st.error("Error: 'model.pkl' matrum 'scaler.pkl' files unga folder-il illai. Check seiyavum!")
    st.stop()

# Title
st.title("🚖 Taxi Trip Pricing Predictor")
st.write("Provide details below to predict the taxi trip price.")
st.markdown("---")

# Screenshot-il ulla padi numerical variables layout
col1, col2 = st.columns(2)

with col1:
    base_fare = st.number_input("Base_Fare", min_value=0.0, value=3.56, step=0.01)
    per_minute_rate = st.number_input("Per_Minute_Rate", min_value=0.0, value=0.32, step=0.01)
    # Trip_Distance_km missing context handled inside backend calculation safely
    trip_distance = st.number_input("Trip_Distance_km", min_value=0.0, value=15.0, step=0.1)
    passenger_count = st.number_input("Passenger_Count", min_value=1.0, value=1.0, step=1.0)

with col2:
    per_km_rate = st.number_input("Per_Km_Rate", min_value=0.0, value=5.00, step=0.01)
    trip_duration = st.number_input("Trip_Duration_Minutes", min_value=0.0, value=5.00, step=0.01)

st.markdown("---")
st.markdown("### 🚦 Environment & Time Settings")

col3, col4 = st.columns(2)

with col3:
    time_of_day = st.selectbox("Time of Day", options=["Morning", "Afternoon", "Evening", "Night"])
    traffic_conditions = st.selectbox("Traffic Conditions", options=["Low", "Medium", "High"])

with col4:
    day_of_week = st.selectbox("Day of Week", options=["Weekday", "Weekend"])
    weather = st.selectbox("Weather", options=["Clear", "Rain", "Snow"])

st.markdown("---")

# Prediction Core Section
if st.button("🔮 Predict Trip Price", use_container_width=True):
    
    # Drop_first mapping calculation (exactly matching the train dummies structure)
    time_evening = 1.0 if time_of_day == "Evening" else 0.0
    time_morning = 1.0 if time_of_day == "Morning" else 0.0
    time_night = 1.0 if time_of_day == "Night" else 0.0
    
    day_weekend = 1.0 if day_of_week == "Weekend" else 0.0
    
    traffic_low = 1.0 if traffic_conditions == "Low" else 0.0
    traffic_medium = 1.0 if traffic_conditions == "Medium" else 0.0
    
    weather_rain = 1.0 if weather == "Rain" else 0.0
    weather_snow = 1.0 if weather == "Snow" else 0.0
    
    # One-to-one mapped list with exact same column sequence as dataset order
    final_features = [
        trip_distance,
        passenger_count,
        base_fare,
        per_km_rate,
        per_minute_rate,
        trip_duration,
        time_evening,
        time_morning,
        time_night,
        day_weekend,
        traffic_low,
        traffic_medium,
        weather_rain,
        weather_snow
    ]
    
    # Setup dimension matrix 
    raw_array = np.array([final_features])
    
    # Apply Standard Scaler
    scaled_data = scaler.transform(raw_array)
    
    # Process output matrix score
    price_output = model.predict(scaled_data)[0]
    price_output = max(0.0, price_output) # Minimum limit check
    
    st.success(f"### Predicted Trip Price: **${price_output:.2f}**")
