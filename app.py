import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Diamond Price Predictor")
st.title("💎 Diamond Price Prediction App")

with open("diamond_price_pipeline.pkl", "rb") as file:
    model = pickle.load(file)

# Inputs
carat = st.number_input("Carat", min_value=0.1)
depth = st.number_input("Depth")
table = st.number_input("Table")
x = st.number_input("X")
y = st.number_input("Y")
z = st.number_input("Z")

cut = st.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
color = st.selectbox("Color", ["D", "E", "F", "G", "H", "I", "J"])
clarity = st.selectbox("Clarity", ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"])

if st.button("Predict Price"):
    input_df = pd.DataFrame([{
        "carat": carat,
        "cut": cut,
        "color": color,
        "clarity": clarity,
        "depth": depth,
        "table": table,
        "x": x,
        "y": y,
        "z": z
    }])

    prediction = model.predict(input_df)[0]
    st.success(f"💰 Estimated Price: ${prediction:,.2f}")