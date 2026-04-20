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
time = st.sidebar.number_input("Time (seconds elapsed)", min_value=0.0, value=50000.0)
amount = st.sidebar.number_input("Transaction Amount (₹)", min_value=0.0, value=100.0)

st.sidebar.markdown("#### Transaction Features (V1-V10)")
v1  = st.sidebar.slider("V1",  -5.0, 5.0, 0.0)
v2  = st.sidebar.slider("V2",  -5.0, 5.0, 0.0)
v3  = st.sidebar.slider("V3",  -5.0, 5.0, 0.0)
v4  = st.sidebar.slider("V4",  -5.0, 5.0, 0.0)
v5  = st.sidebar.slider("V5",  -5.0, 5.0, 0.0)
v6  = st.sidebar.slider("V6",  -5.0, 5.0, 0.0)
v7  = st.sidebar.slider("V7",  -5.0, 5.0, 0.0)
v8  = st.sidebar.slider("V8",  -5.0, 5.0, 0.0)
v9  = st.sidebar.slider("V9",  -5.0, 5.0, 0.0)
v10 = st.sidebar.slider("V10", -5.0, 5.0, 0.0)

st.sidebar.markdown("#### Transaction Features (V11-V28)")
v11 = st.sidebar.slider("V11", -5.0, 5.0, 0.0)
v12 = st.sidebar.slider("V12", -5.0, 5.0, 0.0)
v13 = st.sidebar.slider("V13", -5.0, 5.0, 0.0)
v14 = st.sidebar.slider("V14", -5.0, 5.0, 0.0)
v15 = st.sidebar.slider("V15", -5.0, 5.0, 0.0)
v16 = st.sidebar.slider("V16", -5.0, 5.0, 0.0)
v17 = st.sidebar.slider("V17", -5.0, 5.0, 0.0)
v18 = st.sidebar.slider("V18", -5.0, 5.0, 0.0)
v19 = st.sidebar.slider("V19", -5.0, 5.0, 0.0)
v20 = st.sidebar.slider("V20", -5.0, 5.0, 0.0)
v21 = st.sidebar.slider("V21", -5.0, 5.0, 0.0)
v22 = st.sidebar.slider("V22", -5.0, 5.0, 0.0)
v23 = st.sidebar.slider("V23", -5.0, 5.0, 0.0)
v24 = st.sidebar.slider("V24", -5.0, 5.0, 0.0)
v25 = st.sidebar.slider("V25", -5.0, 5.0, 0.0)
v26 = st.sidebar.slider("V26", -5.0, 5.0, 0.0)
v27 = st.sidebar.slider("V27", -5.0, 5.0, 0.0)
v28 = st.sidebar.slider("V28", -5.0, 5.0, 0.0)

# Predict button
if st.button("🔎 Analyse Transaction"):

    # Scale amount and time
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    amount_scaled = (amount - 66.28) / 50
    time_scaled   = (time - 94000) / 47000

    input_data = np.array([[time_scaled, v1, v2, v3, v4, v5, v6, v7, v8,
                             v9, v10, v11, v12, v13, v14, v15, v16, v17,
                             v18, v19, v20, v21, v22, v23, v24, v25, v26,
                             v27, v28, amount_scaled]])

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
    st.markdown("#### 📌 Top Features Used by Model")
    features = ['V14','V12','V3','V10','V4','V11','V2','V17','V16','V6']
    values   = [v14, v12, v3, v10, v4, v11, v2, v17, v16, v6]
    feat_df  = pd.DataFrame({'Feature': features, 'Value': values})
    st.dataframe(feat_df)

st.markdown("---")
st.markdown("*Built for ABA Final Project | Federal Bank TSM Centre of Excellence*")