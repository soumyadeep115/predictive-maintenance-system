import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load trained model
with open("rf_model.pkl", "rb") as f:
    model = pickle.load(f)

# App title
st.title("Predictive Maintenance Dashboard")
st.write("Predict machine failure based on sensor inputs")

# Sidebar inputs
st.sidebar.header("Machine Sensor Inputs")

air_temp = st.sidebar.number_input(
    "Air Temperature [K]",
    min_value=295.0,
    max_value=305.0,
    value=298.0
)

process_temp = st.sidebar.number_input(
    "Process Temperature [K]",
    min_value=305.0,
    max_value=315.0,
    value=308.0
)

rpm = st.sidebar.number_input(
    "Rotational Speed [rpm]",
    min_value=1100,
    max_value=3000,
    value=1500
)

torque = st.sidebar.number_input(
    "Torque [Nm]",
    min_value=0.0,
    max_value=80.0,
    value=40.0
)

tool_wear = st.sidebar.number_input(
    "Tool Wear [min]",
    min_value=0,
    max_value=250,
    value=50
)

machine_type = st.sidebar.selectbox(
    "Machine Type",
    ["H", "L", "M"]
)

# Encode machine type
type_l = 1 if machine_type == "L" else 0
type_m = 1 if machine_type == "M" else 0

# Prepare input
input_data = np.array([[
    air_temp,
    process_temp,
    rpm,
    torque,
    tool_wear,
    type_l,
    type_m
]])

# Prediction
prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data)[0][1]

# Display prediction result
st.subheader("Prediction Result")

if probability >= 0.75:
    st.error("Critical Failure Risk")
elif probability >= 0.40:
    st.warning("Warning: Potential Failure")
else:
    st.success("Normal Operation")

st.metric("Failure Probability", f"{probability * 100:.2f}%")

# Sensor trend visualization
trend_data = pd.DataFrame({
    'Sensor': [
        'Air Temp',
        'Process Temp',
        'RPM',
        'Torque',
        'Tool Wear'
    ],
    'Value': [
        air_temp,
        process_temp,
        rpm,
        torque,
        tool_wear
    ]
})

st.subheader("Sensor Trends")
st.bar_chart(trend_data.set_index('Sensor'))