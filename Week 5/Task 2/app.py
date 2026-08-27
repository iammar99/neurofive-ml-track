# ==========================================
#        Importing Libraries
# ==========================================

import streamlit as st
import pandas as pd
import joblib


# ==========================================
#          Load Saved Model
# ==========================================

model = joblib.load(
    "telco_churn_pipeline.pkl"
)


# ==========================================
#          Streamlit Page Setup
# ==========================================

st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    page_icon="📊"
)


# ==========================================
#              Title
# ==========================================

st.title("📊 Telco Customer Churn Prediction")

st.write(
    "Enter the customer's information below "
    "to predict whether the customer is likely to churn."
)


# ==========================================
#           Customer Information
# ==========================================

st.header("Customer Information")


tenure = st.number_input(
    "Tenure (months)",
    min_value=0,
    max_value=100,
    value=12
)


monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)


total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=840.0
)


contract = st.selectbox(
    "Contract",
    [
        "Month-to-month",
        "One year",
        "Two year"
    ]
)


payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)


internet_service = st.selectbox(
    "Internet Service",
    [
        "DSL",
        "Fiber optic",
        "No"
    ]
)


online_security = st.selectbox(
    "Online Security",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


tech_support = st.selectbox(
    "Tech Support",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


senior_citizen = st.selectbox(
    "Senior Citizen",
    [
        0,
        1
    ]
)


partner = st.selectbox(
    "Partner",
    [
        "Yes",
        "No"
    ]
)


dependents = st.selectbox(
    "Dependents",
    [
        "Yes",
        "No"
    ]
)


# ==========================================
#        Additional Service Information
# ==========================================

st.header("Services")


phone_service = st.selectbox(
    "Phone Service",
    [
        "Yes",
        "No"
    ]
)


multiple_lines = st.selectbox(
    "Multiple Lines",
    [
        "Yes",
        "No",
        "No phone service"
    ]
)


online_backup = st.selectbox(
    "Online Backup",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


device_protection = st.selectbox(
    "Device Protection",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


streaming_tv = st.selectbox(
    "Streaming TV",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


streaming_movies = st.selectbox(
    "Streaming Movies",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


paperless_billing = st.selectbox(
    "Paperless Billing",
    [
        "Yes",
        "No"
    ]
)


# ==========================================
#       Account Information
# ==========================================

st.header("Account Information")


gender = st.selectbox(
    "Gender",
    [
        "Male",
        "Female"
    ]
)


senior_citizen = st.selectbox(
    "Senior Citizen",
    [
        0,
        1
    ]
)


# ==========================================
#          Prediction Button
# ==========================================

if st.button("🔮 Predict Churn"):

    # ======================================
    #      Create Engineered Features
    # ======================================

    if tenure == 0:
        average_charge = 0
    else:
        average_charge = total_charges / tenure


    if tenure >= 12:
        long_term_customer = 1
    else:
        long_term_customer = 0


    # ======================================
    #          Create Input Data
    # ======================================

    input_data = pd.DataFrame({

        "gender": [gender],

        "SeniorCitizen": [senior_citizen],

        "Partner": [partner],

        "Dependents": [dependents],

        "tenure": [tenure],

        "PhoneService": [phone_service],

        "MultipleLines": [multiple_lines],

        "InternetService": [internet_service],

        "OnlineSecurity": [online_security],

        "OnlineBackup": [online_backup],

        "DeviceProtection": [device_protection],

        "TechSupport": [tech_support],

        "StreamingTV": [streaming_tv],

        "StreamingMovies": [streaming_movies],

        "Contract": [contract],

        "PaperlessBilling": [paperless_billing],

        "PaymentMethod": [payment_method],

        "MonthlyCharges": [monthly_charges],

        "TotalCharges": [total_charges],

        "AverageCharge": [average_charge],

        "LongTermCustomer": [long_term_customer]
    })


    # ======================================
    #             Prediction
    # ======================================

    prediction = model.predict(
        input_data
    )


    # ======================================
    #          Display Prediction
    # ======================================

    if prediction[0] == 1:

        st.error(
            "⚠️ Customer is likely to CHURN."
        )

    else:

        st.success(
            "✅ Customer is likely to STAY."
        )
