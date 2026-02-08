import streamlit as st
import pandas as pd
import pickle

# Load trained model
with open("diamond_knn_model.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(page_title="Diamond Price Predictor")

st.title("💎 Diamond Price Prediction App")
st.write("Enter diamond details to predict the price")

# User inputs
carat = st.number_input("Carat", min_value=0.1, step=0.01)
depth = st.number_input("Depth", step=0.1)
table = st.number_input("Table", step=0.1)
x = st.number_input("X (length)", step=0.1)
y = st.number_input("Y (width)", step=0.1)
z = st.number_input("Z (height)", step=0.1)

cut = st.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
color = st.selectbox("Color", ["D", "E", "F", "G", "H", "I", "J"])
clarity = st.selectbox(
    "Clarity",
    ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
)

# Predict button
if st.button("Predict Price"):
    input_data = pd.DataFrame({
        "carat": [carat],
        "cut": [cut],
        "color": [color],
        "clarity": [clarity],
        "depth": [depth],
        "table": [table],
        "x": [x],
        "y": [y],
        "z": [z]
    })

    prediction = model.predict(input_data)[0]
    st.success(f"💰 Estimated Diamond Price: ${prediction:,.2f}")