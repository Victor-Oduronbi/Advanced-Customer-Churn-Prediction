import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Page Configuration and Styling
st.set_page_config(
    page_title="Telecom Churn Watch",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Telecom Customer Churn Prediction Dashboard")
st.markdown("""
This production-ready dashboard utilizes a high-recall Logistic Regression machine learning pipeline 
to predict customer churn risk and calculate the financial impact of retention strategies in real-time.
""")

# 2. Safely Load Serialized Model Artifacts
@st.cache_resource
def load_pipeline_artifacts():
    try:
        model = joblib.load('champion_model.pkl')
        preprocessor = joblib.load('preprocessor.pkl')
        return model, preprocessor
    except FileNotFoundError:
        st.error("⚠️ Model artifacts missing! Please run your training pipeline script first to generate 'champion_model.pkl' and 'preprocessor.pkl'.")
        return None, None

model, preprocessor = load_pipeline_artifacts()

if model and preprocessor:
    # 3. Sidebar Input Elements for Customer Features
    st.sidebar.header("👤 Customer Profile Input")
    
    # Numerical Inputs
    tenure = st.sidebar.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
    monthly_charges = st.sidebar.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0)
    total_charges = st.sidebar.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=float(tenure * monthly_charges))
    
    # Categorical Inputs (Matching original IBM feature names and cases exactly)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.sidebar.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    partner = st.sidebar.selectbox("Has Partner?", ["Yes", "No"])
    dependents = st.sidebar.selectbox("Has Dependents?", ["Yes", "No"])
    phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
    internet_service = st.sidebar.selectbox("Internet Service Provider", ["DSL", "Fiber optic", "No"])
    online_security = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.sidebar.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])

    # 4. Construct Input DataFrame
    input_data = pd.DataFrame([{
        'gender': gender, 'SeniorCitizen': senior_citizen, 'Partner': partner, 'Dependents': dependents,
        'tenure': tenure, 'PhoneService': phone_service, 'MultipleLines': multiple_lines,
        'InternetService': internet_service, 'OnlineSecurity': online_security, 'OnlineBackup': online_backup,
        'DeviceProtection': device_protection, 'TechSupport': tech_support, 'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies, 'Contract': contract, 'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method, 'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges
    }])

    # 5. Core Operational Prediction Logic
    st.subheader("🔮 Prediction Analysis")
    
    # Process inputs through the saved scikit-learn preprocessor layout
    processed_input = preprocessor.transform(input_data)
    
    # Extract prediction probability and binary class
    churn_probability = model.predict_proba(processed_input)[0][1]
    prediction = model.predict(processed_input)[0]
    
    # Visual Layout Split
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(label="Churn Probability Score", value=f"{churn_probability * 100:.1f}%")
        if prediction == 1:
            st.error("🚨 HIGH RISK: This customer is highly likely to churn!")
        else:
            st.success("✅ LOW RISK: This customer is highly likely to remain loyal.")
            
    with col2:
        # 6. Real-time Corporate Financial Calculator Panel
        st.write("### 💼 Financial ROI Calculator")
        clv = 400.00
        incentive_cost = 40.00
        success_rate = 0.60
        
        if prediction == 1:
            st.markdown(f"""
            **Risk Mitigation Plan Recommendation:**
            * **Action Required:** Disburse a **${incentive_cost:.2f}** retention incentive immediately.
            * **Financial Evaluation:**
                * Cost to do nothing (Let them Churn): **${clv:.2f}**
                * Expected cost with model intervention: **${incentive_cost + ((1 - success_rate) * clv):.2f}**
                * **Net Saved Revenue for this account: ${clv - (incentive_cost + ((1 - success_rate) * clv)):.2f}**
            """)
        else:
            st.markdown(f"""
            **Account Status Maintenance:**
            * **Action Required:** No active retention spending necessary.
            * **Financial Evaluation:** Standard account monitoring. Cost of false proactive incentives avoided: **${incentive_cost:.2f}**
            """)