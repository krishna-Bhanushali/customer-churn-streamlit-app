import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide"
)

# ---------------- LOAD FILES ---------------- #

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# ---------------- TITLE ---------------- #

st.title("📊 Customer Churn Prediction App")
st.caption("AI-powered customer retention system")

# ---------------- SIDEBAR ---------------- #

st.sidebar.header("🧾 Customer Information")

# Numeric Inputs
tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)

monthly_charges = st.sidebar.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=200.0,
    value=70.0
)

# Gender
gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

# Senior Citizen
senior = st.sidebar.selectbox(
    "Senior Citizen",
    ["Yes", "No"]
)

# Partner
partner = st.sidebar.selectbox(
    "Partner",
    ["Yes", "No"]
)

# Dependents
dependents = st.sidebar.selectbox(
    "Dependents",
    ["Yes", "No"]
)

# Phone Service
phone_service = st.sidebar.selectbox(
    "Phone Service",
    ["Yes", "No"]
)

# Internet Service
internet_service = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

# Online Security
online_security = st.sidebar.selectbox(
    "Online Security",
    ["Yes", "No"]
)

# Tech Support
tech_support = st.sidebar.selectbox(
    "Tech Support",
    ["Yes", "No"]
)

# Streaming TV
streaming_tv = st.sidebar.selectbox(
    "Streaming TV",
    ["Yes", "No"]
)

# Contract
contract = st.sidebar.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

# Paperless Billing
paperless = st.sidebar.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

# Payment Method
payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

# Sidebar Metrics
st.sidebar.markdown("---")

st.sidebar.metric(
    "Model Type",
    "Logistic Regression"
)

st.sidebar.metric(
    "Dataset Rows",
    "7043"
)

# ---------------- TABS ---------------- #

tab1, tab2, tab3 = st.tabs(
    ["🔮 Prediction", "📈 Analytics", "ℹ️ About"]
)

# TAB 1 - PREDICTION

with tab1:

    st.subheader("Customer Prediction Dashboard")

    # Create Empty DataFrame
    input_data = pd.DataFrame(columns=columns)

    # Fill row with zeros
    input_data.loc[0] = 0

    # ---------------- NUMERIC FEATURES ---------------- #

    input_data.loc[0, "tenure"] = tenure
    input_data.loc[0, "MonthlyCharges"] = monthly_charges

    # ---------------- BINARY FEATURES ---------------- #

    input_data.loc[0, "gender_Male"] = (
        1 if gender == "Male" else 0
    )

    input_data.loc[0, "SeniorCitizen"] = (
        1 if senior == "Yes" else 0
    )

    input_data.loc[0, "Partner_Yes"] = (
        1 if partner == "Yes" else 0
    )

    input_data.loc[0, "Dependents_Yes"] = (
        1 if dependents == "Yes" else 0
    )

    input_data.loc[0, "PhoneService_Yes"] = (
        1 if phone_service == "Yes" else 0
    )

    input_data.loc[0, "PaperlessBilling_Yes"] = (
        1 if paperless == "Yes" else 0
    )

    input_data.loc[0, "OnlineSecurity_Yes"] = (
        1 if online_security == "Yes" else 0
    )

    input_data.loc[0, "TechSupport_Yes"] = (
        1 if tech_support == "Yes" else 0
    )

    input_data.loc[0, "StreamingTV_Yes"] = (
        1 if streaming_tv == "Yes" else 0
    )

    # ---------------- INTERNET SERVICE ---------------- #

    if internet_service == "Fiber optic":
        input_data.loc[0, "InternetService_Fiber optic"] = 1

    elif internet_service == "No":
        input_data.loc[0, "InternetService_No"] = 1

    # DSL becomes default (0)

    # ---------------- CONTRACT ---------------- #

    if contract == "One year":
        input_data.loc[0, "Contract_One year"] = 1

    elif contract == "Two year":
        input_data.loc[0, "Contract_Two year"] = 1

    # Month-to-month becomes default (0)

    # ---------------- PAYMENT METHOD ---------------- #

    if payment_method == "Electronic check":
        input_data.loc[0, "PaymentMethod_Electronic check"] = 1

    elif payment_method == "Mailed check":
        input_data.loc[0, "PaymentMethod_Mailed check"] = 1

    elif payment_method == "Credit card (automatic)":
        input_data.loc[0, "PaymentMethod_Credit card (automatic)"] = 1

    # Bank transfer becomes default (0)

    # ---------------- SCALE INPUT ---------------- #

    input_scaled = scaler.transform(input_data)

    # ---------------- PREDICT BUTTON ---------------- #

    if st.button("Predict Churn"):

        prediction = model.predict(input_scaled)

        probability = model.predict_proba(
            input_scaled
        )[0][1]

        # ---------------- METRICS ---------------- #

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Churn Probability",
                f"{round(probability * 100, 2)}%"
            )

        with col2:
            st.metric(
                "Monthly Charges",
                f"₹{monthly_charges}"
            )

        with col3:
            st.metric(
                "Tenure",
                f"{tenure} Months"
            )

        # ---------------- PROGRESS BAR ---------------- #

        st.progress(int(probability * 100))

        # ---------------- RESULT ---------------- #

        st.subheader("Prediction Result")

        if prediction[0] == 1:
            st.error(
                "⚠️ High risk of customer churn detected"
            )

        else:
            st.success(
                "✅ Customer is likely to stay"
            )

        # ---------------- RISK LEVEL ---------------- #

        st.subheader("Risk Analysis")

        if probability < 0.3:
            st.success(f"Low Churn Probability ~{int(probability * 100)}%")

        elif probability < 0.7:
            st.warning(f"Medium Churn Probability ~{int(probability * 100)}%")

        else:
            st.error(f"High Churn Probability ~{int(probability * 100)}%")

        # ---------------- CUSTOMER SUMMARY ---------------- #

        st.subheader("📋 Customer Summary")

        st.write(f"Gender: {gender}")
        st.write(f"Contract Type: {contract}")
        st.write(f"Internet Service: {internet_service}")
        st.write(f"Payment Method: {payment_method}")
        st.write(f"Monthly Charges: ₹{monthly_charges}")
        st.write(f"Tenure: {tenure} months")

        # ---------------- BAR CHART ---------------- #

        chart_data = pd.DataFrame({
            "Category": ["Stay", "Churn"],
            "Probability": [
                1 - probability,
                probability
            ]
        })

        st.subheader("📊 Probability Comparison")

        st.bar_chart(
            chart_data.set_index("Category")
        )

# =========================================================
# TAB 2 - ANALYTICS
# =========================================================

with tab2:

    st.subheader("📈 Analytics Dashboard")

    analytics_data = pd.DataFrame({
        "Metric": [
            "Model Accuracy",
            "Dataset Size",
            "Features Used"
        ],
        "Value": [
            "82%",
            "7043",
            len(columns)
        ]
    })

    st.dataframe(analytics_data)

    # Dummy chart
    with st.container(border=True):
        fig, ax = plt.subplots(figsize=[4,2.5])

        ax.bar(
            ["Stay", "Churn"],
            [70, 30]
        )

        ax.set_title("Sample Churn Distribution")

        plt.tight_layout()

        st.pyplot(fig, use_container_width=False)

# =========================================================
# TAB 3 - ABOUT
# =========================================================

with tab3:

    st.subheader("ℹ️ About This Project")

    st.write("""
    This application predicts customer churn using
    Machine Learning.

    The model was trained using the IBM Telco
    Customer Churn dataset.

    Technologies Used:
    - Python
    - Streamlit
    - Pandas
    - Scikit-learn
    - Logistic Regression
    """)

    with st.expander("How does this model work?"):

        st.write("""
        The model analyzes customer information such as:

        - Tenure
        - Monthly Charges
        - Contract Type
        - Internet Service
        - Payment Method
        - Tech Support

        and predicts the probability of customer churn.
        """)

# ---------------- FOOTER ---------------- #

st.markdown("---")

st.write(
    "Built by Krishna Bhanushali using Streamlit & Machine Learning 🚀"
)