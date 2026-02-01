import streamlit as st
import joblib
import numpy as np 
import os

# Set page config for a better browser tab title
st.set_page_config(page_title="House Price Predictor", page_icon="🏠")

# Load model with caching to improve performance
@st.cache_resource
def load_model():
    # Ensure the path matches your folder structure
    model_path = os.path.join("models", "model.pkl")
    return joblib.load(model_path)

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'model.pkl' is in the 'models' folder.")
    st.stop()

st.title("🏠 House Price Prediction App")
st.markdown("### Powered by Machine Learning")
st.write("This app uses a Random Forest Regressor to predict house prices based on specific features. Adjust the values below to get a prediction.")

st.divider()

# Input columns for better layout
col1, col2 = st.columns(2)

with col1:
    bedrooms = st.number_input("Number of bedrooms", min_value=0, value=3)
    bathrooms = st.number_input("Number of bathrooms", min_value=0.0, value=2.0, step=0.5)
    condition = st.slider("Condition of the house (1-5)", min_value=1, max_value=5, value=3)

with col2:
    livingarea = st.number_input("Living area (sq ft)", min_value=0, value=2000)
    numberofschools = st.number_input("Number of schools nearby", min_value=0, value=2)

st.divider()

# Prepare input data
input_data = [[bedrooms, bathrooms, livingarea, condition, numberofschools]]

if st.button("Predict Price", type="primary"):
    # Convert input to array
    x_array = np.array(input_data)
    
    # Predict
    prediction = model.predict(x_array)[0]
    
    # Display result
    st.balloons()
    st.success(f"Estimated House Price: **${prediction:,.2f}**")