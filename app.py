import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load saved model
with open("taxi_price_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Taxi Price Predictor", page_icon="🚖")

st.title("🚖 Taxi Trip Price Prediction App")
st.write("Enter trip details below:")

# -------------------------
# Numeric Inputs
# -------------------------
col1, col2 = st.columns(2)

with col1:
    trip_distance = st.number_input("Trip Distance (km)", min_value=0.0)
    trip_duration = st.number_input("Trip Duration (minutes)", min_value=0.0)
    passenger_count = st.number_input("Passenger Count", min_value=1, step=1)

with col2:
    base_fare = st.number_input("Base Fare", min_value=0.0)
    per_km_rate = st.number_input("Per Km Rate", min_value=0.0)
    per_minute_rate = st.number_input("Per Minute Rate", min_value=0.0)

# -------------------------
# Categorical Inputs
# -------------------------
st.subheader("Trip Conditions")

time_of_day = st.selectbox(
    "Time of Day",
    ["Morning", "Afternoon", "Evening", "Night"]
)

day_of_week = st.selectbox(
    "Day of Week",
    ["Monday", "Tuesday", "Wednesday", "Thursday",
     "Friday", "Saturday", "Sunday"]
)

traffic = st.selectbox(
    "Traffic Conditions",
    ["Low", "Medium", "High"]
)

weather = st.selectbox(
    "Weather",
    ["Clear", "Rainy", "Foggy", "Stormy"]
)

# -------------------------
# Create Input DataFrame
# -------------------------
input_data = pd.DataFrame([{
    "Trip_Distance_km": np.log1p(trip_distance),  # same transform as training
    "Trip_Duration_Minutes": trip_duration,
    "Passenger_Count": passenger_count,
    "Base_Fare": base_fare,
    "Per_Km_Rate": per_km_rate,
    "Per_Minute_Rate": per_minute_rate,
    "Time_of_Day": time_of_day,
    "Day_of_Week": day_of_week,
    "Traffic_Conditions": traffic,
    "Weather": weather
}])

# -------------------------
# Prediction
# -------------------------
if st.button("Predict Price"):

    prediction_log = model.predict(input_data)[0]

    # Reverse log transformation
    predicted_price = np.expm1(prediction_log)

    st.subheader("Estimated Trip Price")
    st.success(f"💰 ${predicted_price:.2f}")

    st.progress(min(predicted_price / 100, 1.0))
