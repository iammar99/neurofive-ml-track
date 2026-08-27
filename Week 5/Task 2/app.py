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
    page_title="Telco Customer Churn",
    page_icon="📊"
)


# ==========================================
#              Title
# ==========================================

st.title("📊 Telco Customer Churn Prediction")

st.write(
    "Enter customer information and click "
    "Predict Churn to see the prediction."
)


# ==========================================
#        Customer Information
# ==========================================

st.header("Customer Information")


gender = st.selectbox(
    "Gender",
    [
        "Male",
        "Female"
    ],
    key="gender"
)


senior_citizen = st.selectbox(
    "Senior Citizen",
    [
        0,
        1
    ],
    key="senior_citizen"
)


partner = st.selectbox(
    "Partner",
    [
        "Yes",
        "No"
    ],
    key="partner"
)


dependents = st.selectbox(
    "Dependents",
    [
        "Yes",
        "No"
    ],
    key="dependents"
)


tenure = st.number_input(
    "Tenure (months)",
    min_value=0,
    max_value=100,
    value=12,
    key="tenure"
)


phone_service = st.selectbox(
    "Phone Service",
    [
        "Yes",
        "No"
    ],
    key="phone_service"
)


multiple_lines = st.selectbox(
    "Multiple Lines",
    [
        "Yes",
        "No",
        "No phone service"
    ],
    key="multiple_lines"
)


internet_service = st.selectbox(
    "Internet Service",
    [
        "DSL",
        "Fiber optic",
        "No"
    ],
    key="internet_service"
)


online_security = st.selectbox(
    "Online Security",
    [
        "Yes",
        "No",
        "No internet service"
    ],
    key="online_security"
)


online_backup = st.selectbox(
    "Online Backup",
    [
        "Yes",
        "No",
        "No internet service"
    ],
    key="online_backup"
)


device_protection = st.selectbox(
    "Device Protection",
    [
        "Yes",
        "No",
        "No internet service"
    ],
    key="device_protection"
)


tech_support = st.selectbox(
    "Tech Support",
    [
        "Yes",
        "No",
        "No internet service"
    ],
    key="tech_support"
)


streaming_tv = st.selectbox(
    "Streaming TV",
    [
        "Yes",
        "No",
        "No internet service"
    ],
    key="streaming_tv"
)


streaming_movies = st.selectbox(
    "Streaming Movies",
    [
        "Yes",
        "No",
        "No internet service"
    ],
    key="streaming_movies"
)


# ==========================================
#          Account Information
# ==========================================

st.header("Account Information")


contract = st.selectbox(
    "Contract",
    [
        "Month-to-month",
        "One year",
        "Two year"
    ],
    key="contract"
)


paperless_billing = st.selectbox(
    "Paperless Billing",
    [
        "Yes",
        "No"
    ],
    key="paperless_billing"
)


payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ],
    key="payment_method"
)


monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0,
    key="monthly_charges"
)


total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=840.0,
    key="total_charges"
)


# ==========================================
#          Prediction Button
# ==========================================

if st.button(
    "🔮 Predict Churn",
    key="predict_button"
):

    # ======================================
    #      Create Engineered Features
    # ======================================

    if tenure == 0:

        average_charge = 0

    else:

        average_charge = (
            total_charges / tenure
        )


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
    #          Display Result
    # ======================================

    st.header("Prediction")


    if prediction[0] == 1:

        st.error(
            "⚠️ Customer is likely to CHURN."
        )

    else:

        st.success(
            "✅ Customer is likely to STAY."
        )
