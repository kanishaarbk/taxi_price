import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Page Configuration
st.set_page_config(page_title="Student Performance Predictor", layout="centered")

# Saved files-ai load seiyya function
@st.cache_resource
def load_assets():
    with open('model.pkl', 'rb') as m_file:
        model = pickle.load(m_file)
    with open('scaler.pkl', 'rb') as s_file:
        scaler = pickle.load(s_file)
    with open('columns.pkl', 'rb') as c_file:
        columns = pickle.load(c_file)
    return model, scaler, columns

try:
    model, scaler, feature_columns = load_assets()
except FileNotFoundError:
    st.error("Error: 'model.pkl', 'scaler.pkl', or 'columns.pkl' file unga folder-il illai. Check seiyavum!")
    st.stop()

# Application Title
st.title("🎓 Student Performance Index Predictor")
st.write("Enter the required details to predict the Performance Index.")
st.markdown("---")

# Dictionary to hold user inputs dynamically based on training features
user_inputs = {}

# inputs-ai rendu column-aga uruvakka split seigirom
col1, col2 = st.columns(2)

for i, col in enumerate(feature_columns):
    # Determine which column to place the widget
    current_col = col1 if i % 2 == 0 else col2
    
    with current_col:
        # Check if the feature is Extracurricular Activities (which needs mapping)
        if col == 'Extracurricular Activities':
            choice = st.selectbox("Extracurricular Activities", options=["Yes", "No"])
            user_inputs[col] = 1 if choice == "Yes" else 0
        else:
            # Map default values or steps based on feature name roughly
            if 'Score' in col:
                user_inputs[col] = st.number_input(f"{col}", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
            elif 'Hour' in col:
                user_inputs[col] = st.number_input(f"{col}", min_value=0.0, max_value=24.0, value=6.0, step=0.5)
            else:
                user_inputs[col] = st.number_input(f"{col}", min_value=0.0, value=5.0, step=1.0)

st.markdown("---")

# Predict Button
if st.button("🔮 Predict Performance Index", use_container_width=True):
    # Convert inputs to DataFrame with exactly the same column structure and sequence as trained
    input_df = pd.DataFrame([user_inputs], columns=feature_columns)
    
    # Preprocessing using saved Scaler
    scaled_features = scaler.transform(input_df)
    
    # Model prediction
    prediction = model.predict(scaled_features)[0]
    
    # Output bounding (0 to 100 limit indices-ku mattum)
    prediction = max(0.0, min(100.0, prediction))
    
    # Display Result
    st.success(f"### Predicted Performance Index: **{prediction:.2f}**")
