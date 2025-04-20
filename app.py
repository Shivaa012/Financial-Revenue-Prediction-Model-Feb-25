import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Title and Styling
st.title(" Company Revenue Predictor")
st.markdown("""
    <style>
        .header {
            text-align: center;
            font-size: 36px;
            color: #4CAF50;
            font-weight: bold;
        }
        .footer {
            font-size: 16px;
            text-align: center;
            margin-top: 20px;
        }
        .linkedin {
            color: #0e76a8;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

features = [
    "grossProfit", "operatingIncome", "costOfRevenue", "totalOperatingExpenses",
    "netIncome", "incomeBeforeTax", "netIncomeFromContinuingOps", "interestExpense"
]

# Input fields
st.subheader("Enter the financial features:")
inputs = []
for feature in features:
    val = st.number_input(f"{feature}", min_value=-1_000_000_000.0, max_value=1_000_000_000.0, step=1.0, value=0.0)
    inputs.append(val)

# Button for Prediction
if st.button("Predict Revenue"):
    try:
        input_data = np.array([inputs])
        input_scaled = scaler.transform(input_data)
        pred_log = model.predict(input_scaled)[0]
        prediction = np.expm1(pred_log)
        st.success(f"💰 **Predicted Revenue**: ${prediction:,.2f}")
    except Exception as e:
        st.error(f"Error: {e}")

# Footer with name and LinkedIn link
st.markdown("""
    <div class="footer">
        Created by: <b>Shivam Sawale</b><br>
        <a class="linkedin" href="https://www.linkedin.com/in/shivam-sawale-350a2b26b" target="_blank">
            LinkedIn: www.linkedin.com/in/shivam-sawale-350a2b26b
        </a>
    </div>
""", unsafe_allow_html=True)
