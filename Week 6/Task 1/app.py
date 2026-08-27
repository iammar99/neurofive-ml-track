# ==========================================
#       House Price Prediction App
# ==========================================

import streamlit as st
import pandas as pd
import joblib


# ==========================================
#            Load Saved Model
# ==========================================

model = joblib.load(
    "house_price_model.pkl"
)


# ==========================================
#          Page Configuration
# ==========================================

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)


# ==========================================
#             Custom Styling
# ==========================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #0f172a;
    }

    .main-title {
        font-size: 42px;
        font-weight: bold;
        text-align: center;
        color: #f7d27a;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 17px;
        margin-bottom: 30px;
    }

    .result-box {
        background-color: #1e293b;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin-top: 25px;
    }

    .result-title {
        color: #94a3b8;
        font-size: 16px;
    }

    .result-price {
        color: #f7d27a;
        font-size: 38px;
        font-weight: bold;
    }

    .info-box {
        background-color: #1e293b;
        padding: 20px;
        border-radius: 12px;
        margin-top: 20px;
    }

    .info-title {
        color: #f7d27a;
        font-size: 18px;
        font-weight: bold;
    }

    .info-text {
        color: #cbd5e1;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================
#                Header
# ==========================================

st.markdown(
    '<div class="main-title">🏠 House Price Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Predict the estimated value of a house using Machine Learning</div>',
    unsafe_allow_html=True
)


# ==========================================
#              Main Columns
# ==========================================

left, right = st.columns(2)


# ==========================================
#          PROPERTY INFORMATION
# ==========================================

with left:

    st.header("🏡 Property Information")


    area = st.number_input(
        "Area",
        min_value=100,
        value=5000,
        step=100
    )


    bedrooms = st.number_input(
        "Bedrooms",
        min_value=1,
        value=3,
        step=1
    )


    bathrooms = st.number_input(
        "Bathrooms",
        min_value=1,
        value=2,
        step=1
    )


    stories = st.number_input(
        "Stories",
        min_value=1,
        value=2,
        step=1
    )


    parking = st.number_input(
        "Parking Spaces",
        min_value=0,
        value=1,
        step=1
    )


    furnishingstatus = st.selectbox(
        "Furnishing Status",
        [
            "furnished",
            "semi-furnished",
            "unfurnished"
        ],
        key="furnishingstatus"
    )


# ==========================================
#           PROPERTY FEATURES
# ==========================================

with right:

    st.header("✨ Property Features")


    mainroad = st.selectbox(
        "Main Road",
        ["yes", "no"],
        key="mainroad"
    )


    guestroom = st.selectbox(
        "Guest Room",
        ["yes", "no"],
        key="guestroom"
    )


    basement = st.selectbox(
        "Basement",
        ["yes", "no"],
        key="basement"
    )


    hotwaterheating = st.selectbox(
        "Hot Water Heating",
        ["yes", "no"],
        key="hotwaterheating"
    )


    airconditioning = st.selectbox(
        "Air Conditioning",
        ["yes", "no"],
        key="airconditioning"
    )


    prefarea = st.selectbox(
        "Preferred Area",
        ["yes", "no"],
        key="prefarea"
    )


# ==========================================
#        Feature Engineering
# ==========================================

area_per_bedroom = (
    area / bedrooms
)


total_rooms = (
    bedrooms + bathrooms
)


# ==========================================
#          Prediction Button
# ==========================================

st.write("")


if st.button(
    "🏠 Predict House Price",
    use_container_width=True
):


    # ======================================
    #          Create Input Data
    # ======================================

    input_data = pd.DataFrame({

        "area": [area],

        "bedrooms": [bedrooms],

        "bathrooms": [bathrooms],

        "stories": [stories],

        "mainroad": [mainroad],

        "guestroom": [guestroom],

        "basement": [basement],

        "hotwaterheating": [
            hotwaterheating
        ],

        "airconditioning": [
            airconditioning
        ],

        "parking": [parking],

        "prefarea": [prefarea],

        "furnishingstatus": [
            furnishingstatus
        ],

        "area_per_bedroom": [
            area_per_bedroom
        ],

        "total_rooms": [
            total_rooms
        ]

    })


    # ======================================
    #             Prediction
    # ======================================

    prediction = model.predict(
        input_data
    )


    predicted_price = prediction[0]


    # ======================================
    #          Display Result
    # ======================================

    st.markdown(
        f"""
        <div class="result-box">

        <div class="result-title">
        ESTIMATED HOUSE PRICE
        </div>

        <div class="result-price">
        Rs. {predicted_price:,.0f}
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # ======================================
    #          House Summary
    # ======================================

    st.write("")

    st.subheader("📋 House Summary")


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Area",
            f"{area:,}"
        )


    with col2:

        st.metric(
            "Bedrooms",
            bedrooms
        )


    with col3:

        st.metric(
            "Bathrooms",
            bathrooms
        )


    with col4:

        st.metric(
            "Parking",
            parking
        )


# ==========================================
#                Footer
# ==========================================

st.write("")

st.markdown(
    """
    <div style="
        text-align:center;
        color:#64748b;
        padding:20px;
    ">

    House Price Prediction • Machine Learning Project

    <br>

    Built with Python, Scikit-learn & Streamlit

    </div>
    """,
    unsafe_allow_html=True
)
