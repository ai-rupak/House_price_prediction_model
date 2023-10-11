import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained machine learning model
with open('house_price_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)


# Define a function to predict house prices
def predict_price(crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat):
    input_data = np.array([crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat]).reshape(1, -1)
    prediction = loaded_model.predict(input_data)
    return prediction[0]


# Set page title and favicon
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏡",
)

# Define CSS style
st.markdown(
    """
    <style>
    body {
        background-color: #f9f9f9;
        font-family: Arial, sans-serif;
    }
    .stApp {
        max-width: 800px;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        background-color: #ffffff;
        margin: auto;
        margin-top: 30px;
    }
    .stButton {
        background-color: #007BFF;
        color: red;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        cursor: pointer;
        font-weight: bold;
        font-size: 16px;
    }
    .stButton:hover {
        background-color: #0056b3;
    }
    h1 {
        color: #007BFF;
        font-size: 28px;
        text-align: center;
        margin-bottom: 20px;
    }
    h2 {
        color: #333333;
        font-size: 24px;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Streamlit UI
st.title("House Price Prediction")
st.write("Enter the following parameters to predict the house price:")



st.header("Input Parameters")
crim = st.number_input("CRIM (Crime rate):", min_value=0.0, max_value=100.0, value=0.0)
zn = st.number_input("ZN (Proportion of residential land zoned for large lots):", min_value=0.0, max_value=100.0, value=0.0)
indus = st.number_input("INDUS (Proportion of non-retail business acres per town):", min_value=0.0, max_value=100.0, value=0.0)
chas = st.selectbox("CHAS (Charles River dummy variable):", [0, 1])
nox = st.number_input("NOX (Nitrogen oxide concentration):", min_value=0.0, max_value=1.0, value=0.0)
rm = st.number_input("RM (Average number of rooms per dwelling):", min_value=0.0, max_value=10.0, value=0.0)
age = st.number_input("AGE (Proportion of owner-occupied units built before 1940):", min_value=0.0, max_value=100.0, value=0.0)
dis = st.number_input("DIS (Weighted distance to employment centers):", min_value=0.0, max_value=10.0, value=0.0)
rad = st.number_input("RAD (Index of accessibility to radial highways):", min_value=0, max_value=24, value=0)
tax = st.number_input("TAX (Property tax rate):", min_value=0, max_value=1000, value=0)
ptratio = st.number_input("PTRATIO (Pupil-teacher ratio):", min_value=0.0, max_value=50.0, value=0.0)
b = st.number_input("B (Proportion of residents of African American descent):", min_value=0.0, max_value=100.0, value=0.0)
lstat = st.number_input("LSTAT (Percentage of lower status population):", min_value=0.0, max_value=100.0, value=0.0)

if st.button("Predict"):
    result = predict_price(crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat)
    st.write(f"Predicted Price: ${result:.2f}")
