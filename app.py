import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model
model = joblib.load('fraud_model.pkl')

# Page config
st.set_page_config(page_title="Fraud Detection System", page_icon="🔍", layout="centered")

st.title("🔍 Credit Card Fraud Detection")
st.markdown("### Real-Time Transaction Risk Scoring System")
st.markdown("---")

st.sidebar.header("Enter Transaction Details")

# Input fields
time   = st.sidebar.number_input("Time (seconds elapsed)", min_value=0.0, value=50000.0)
amount = st.sidebar.number_input("Transaction Amount (₹)", min_value=0.0, value=100.0)

st.sidebar.markdown("#### Transaction Behaviour Inputs")
v14 = st.sidebar.slider("Authentication Pattern",      -20.0, 20.0, 0.0)
v12 = st.sidebar.slider("Merchant Category Pattern",   -20.0, 20.0, 0.0)
v3  = st.sidebar.slider("Transaction Frequency Pattern",-20.0, 20.0, 0.0)
v10 = st.sidebar.slider("Geographic Location Pattern", -20.0, 20.0, 0.0)
v4  = st.sidebar.slider("Spending Behaviour Pattern",  -20.0, 20.0, 0.0)
v11 = st.sidebar.slider("Card Usage History Pattern",  -20.0, 20.0, 0.0)
v2  = st.sidebar.slider("Transaction Time Pattern",    -20.0, 20.0, 0.0)
v17 = st.sidebar.slider("Device & Channel Pattern",    -20.0, 20.0, 0.0)

# Predict button
if st.button("🔎 Analyse Transaction"):

    amount_scaled = (amount - 66.28) / 50
    time_scaled   = (time - 94000) / 47000

    # All 30 features — only top 8 are user inputs, rest set to 0
    input_data = np.array([[time_scaled,
                             0.0,   # V1
                             v2,    # V2  - Transaction Time Pattern
                             v3,    # V3  - Transaction Frequency Pattern
                             v4,    # V4  - Spending Behaviour Pattern
                             0.0,   # V5
                             0.0,   # V6
                             0.0,   # V7
                             0.0,   # V8
                             0.0,   # V9
                             v10,   # V10 - Geographic Location Pattern
                             v11,   # V11 - Card Usage History Pattern
                             v12,   # V12 - Merchant Category Pattern
                             0.0,   # V13
                             v14,   # V14 - Authentication Pattern
                             0.0,   # V15
                             0.0,   # V16
                             v17,   # V17 - Device & Channel Pattern
                             0.0,   # V18
                             0.0,   # V19
                             0.0,   # V20
                             0.0,   # V21
                             0.0,   # V22
                             0.0,   # V23
                             0.0,   # V24
                             0.0,   # V25
                             0.0,   # V26
                             0.0,   # V27
                             0.0,   # V28
                             amount_scaled]])

    prob  = model.predict_proba(input_data)[0][1]
    score = round(prob * 100, 2)

    st.markdown("---")
    st.markdown("## 📊 Risk Assessment Result")

    col1, col2 = st.columns(2)
    col1.metric("Fraud Probability", f"{prob:.4f}")
    col2.metric("Risk Score", f"{score} / 100")

    if score >= 70:
        st.error("🔴 HIGH RISK — This transaction is likely FRAUDULENT!")
    elif score >= 40:
        st.warning("🟡 MEDIUM RISK — This transaction needs further review.")
    else:
        st.success("🟢 LOW RISK — This transaction appears LEGITIMATE.")

    st.markdown("---")
    st.markdown("#### 📌 Feature Analysis")
    feat_df = pd.DataFrame({
        'Feature'    : ['Authentication Pattern',
                        'Merchant Category Pattern',
                        'Transaction Frequency Pattern',
                        'Geographic Location Pattern',
                        'Spending Behaviour Pattern',
                        'Card Usage History Pattern',
                        'Transaction Time Pattern',
                        'Device & Channel Pattern'],
        'Value'      : [v14, v12, v3, v10, v4, v11, v2, v17],
        'Importance' : ['20.05%','14.00%','10.70%',
                        '8.88%','8.67%','8.04%','7.54%','7.03%']
    })
    st.dataframe(feat_df)

st.markdown("---")
st.markdown("*Built for ABA Final Project | Federal Bank TSM Centre of Excellence*")
