import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Page Configuration
st.set_page_config(page_title="Credit Risk Prediction App", page_icon="💳", layout="centered")

st.title("💳 Credit Risk Prediction System")
st.write("College Project - Enter customer details below to check loan default risk.")

# Model Training (Caching taaki baar-baar train na ho)
@st.cache_resource
def load_and_train_model():
    df = pd.read_csv("bankloans.csv")
    df = df.dropna()
    
    # 5 Main Features select kar rahe hain jo dataset mein available hote hain
    # (Aap apne dataset ke columns ke mutabiq inke naam check kar sakte hain)
    selected_features = ['age', 'ed', 'employ', 'address', 'debtinc']
    
    X = df[selected_features]
    y = df['default']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression()
    model.fit(X_scaled, y)
    
    return model, scaler, selected_features

model, scaler, features = load_and_train_model()

# User Interface - 5 Input Fields
st.subheader("📝 Enter Customer Details:")

# 5 Inputs ke liye fields
age = st.number_input("Age (Umr)", min_value=18, max_value=100, value=30)
ed = st.number_input("Education Level (Years of Education)", min_value=1, max_value=20, value=12)
employ = st.number_input("Years at Current Employer (Job Experience)", min_value=0, max_value=50, value=5)
address = st.number_input("Years at Current Address", min_value=0, max_value=50, value=3)
debtinc = st.number_input("Debt to Income Ratio (%)", min_value=0.0, max_value=100.0, value=10.0)

# Prediction Button
if st.button("Predict Credit Risk"):
    # Input data ko array mein convert karna
    input_data = np.array([[age, ed, employ, address, debtinc]])
    input_scaled = scaler.transform(input_data)
    
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)
    
    st.markdown("---")
    st.subheader("📊 Prediction Result:")
    
    if prediction[0] == 1:
        st.error(f"⚠️ **High Risk!** Customer is likely to default on the loan. (Probability: {prediction_proba[0][1]*100:.2f}%)")
    else:
        st.success(f"✅ **Low Risk!** Customer is safe and likely to repay the loan. (Probability: {prediction_proba[0][0]*100:.2f}%)")