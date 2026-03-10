import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from xgboost import XGBClassifier

# ------------------------------------------------
# Page Configuration
# ------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------
# Enhanced Custom Styling
# ------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 48px;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 20px 0;
    }
    .sub-header {
        text-align: center;
        color: #a0aec0;
        font-size: 18px;
        margin-bottom: 30px;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 15px;
        border-radius: 10px;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# Load Model & Files
# ------------------------------------------------
model = XGBClassifier()
model.load_model("churn_model.json")

scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

# ------------------------------------------------
# Header Section
# ------------------------------------------------
st.markdown("<div class='main-header'>📊 Customer Churn Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>AI-Powered Telecom Customer Churn Prediction System</div>", unsafe_allow_html=True)
st.markdown("---")

# ------------------------------------------------
# Sidebar Navigation
# ------------------------------------------------
st.sidebar.title("🎯 Navigation")
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Select Mode:",
    ["🔍 Single Prediction", "📂 Bulk Prediction", "📊 Model Insights"]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Use Single Prediction for individual customers or Bulk Prediction for datasets.")

# ======================================================
# 🔍 SINGLE CUSTOMER PREDICTION
# ======================================================
if menu == "🔍 Single Prediction":

    st.markdown("### 👤 Enter Customer Details")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📋 Demographics**")
        gender = st.selectbox("Gender", ["Male", "Female"])
        partner = st.selectbox("Has Partner?", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
        paperless_billing = st.selectbox("Paperless Billing?", ["Yes", "No"])
        contract = st.selectbox("Contract Type",
                                ["Month-to-month", "One year", "Two year"])

    with col2:
        st.markdown("**💰 Financial & Service Details**")
        tenure = st.number_input("Tenure (Months)", min_value=0, value=12)
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
        total_charges = st.number_input("Total Charges", min_value=0.0, value=1000.0)
        internet_service = st.selectbox("Internet Service",
                                        ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security?", ["Yes", "No"])
        tech_support = st.selectbox("Tech Support?", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method",
                                      ["Electronic check",
                                       "Mailed check",
                                       "Bank transfer (automatic)",
                                       "Credit card (automatic)"])

    st.markdown("---")

    if st.button("🚀 Predict Churn Risk"):

        input_data = {
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "gender": gender,
            "Partner": partner,
            "Dependents": dependents,
            "PaperlessBilling": paperless_billing,
            "Contract": contract,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "TechSupport": tech_support,
            "PaymentMethod": payment_method
        }

        input_df = pd.DataFrame([input_data])
        input_df = pd.get_dummies(input_df, drop_first=True)
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)

        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

        prediction = model.predict(input_df.values)
        probability = model.predict_proba(input_df.values)[0][1]

        st.markdown("---")
        st.markdown("### 🎯 Prediction Result")

        # Gauge Chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=probability*100,
            title={"text": "Churn Risk %", "font": {"size": 24}},
            delta={"reference": 50},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkred"},
                "steps": [
                    {"range": [0, 30], "color": "lightgreen"},
                    {"range": [30, 70], "color": "orange"},
                    {"range": [70, 100], "color": "red"}
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 70
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        # Metrics
        colA, colB, colC = st.columns(3)

        with colA:
            st.metric("Churn Probability", f"{probability*100:.1f}%")

        with colB:
            risk_level = "Low" if probability < 0.3 else "Medium" if probability < 0.7 else "High"
            st.metric("Risk Level", risk_level)

        with colC:
            retention_score = int((1-probability)*100)
            st.metric("Retention Score", f"{retention_score}%")

        st.markdown("---")

        if probability < 0.3:
            st.success("✅ **Low Risk Customer** - Likely to stay with the service")
        elif probability < 0.7:
            st.warning("⚠️ **Medium Risk Customer** - Consider retention strategies")
        else:
            st.error("🚨 **High Churn Risk** - Immediate action recommended!")

# ======================================================
# 📂 BULK PREDICTION
# ======================================================
elif menu == "📂 Bulk Prediction":

    st.markdown("### 📤 Upload Customer Dataset")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:

        with st.spinner("Processing data..."):
            data = pd.read_csv(uploaded_file)

            st.success(f"✅ Loaded {len(data)} customer records")

            data_encoded = pd.get_dummies(data, drop_first=True)
            data_encoded = data_encoded.reindex(columns=feature_columns, fill_value=0)
            data_encoded[numeric_cols] = scaler.transform(data_encoded[numeric_cols])

            predictions = model.predict(data_encoded.values)
            probabilities = model.predict_proba(data_encoded.values)[:, 1]

            data["Churn_Prediction"] = predictions
            data["Churn_Probability"] = probabilities

        # Summary Metrics
        st.markdown("### 📊 Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Customers", len(data))
        with col2:
            churn_count = data["Churn_Prediction"].sum()
            st.metric("Predicted Churns", churn_count)
        with col3:
            churn_rate = (churn_count/len(data))*100
            st.metric("Churn Rate", f"{churn_rate:.1f}%")
        with col4:
            avg_prob = data["Churn_Probability"].mean()*100
            st.metric("Avg Risk", f"{avg_prob:.1f}%")

        st.markdown("---")

        # Risk Distribution Chart
        st.markdown("### 📈 Churn Risk Distribution")
        fig = px.histogram(
            data,
            x="Churn_Probability",
            nbins=30,
            title="Customer Churn Risk Distribution",
            labels={"Churn_Probability": "Churn Probability"},
            color_discrete_sequence=["#667eea"]
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📋 Detailed Results")
        st.dataframe(data, use_container_width=True)

        csv = data.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Predictions CSV",
            csv,
            "churn_predictions.csv",
            "text/csv"
        )

# ======================================================
# 📊 MODEL INSIGHTS
# ======================================================
elif menu == "📊 Model Insights":

    st.markdown("### 🔬 Feature Importance Analysis")

    importance = model.feature_importances_

    feature_importance = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False).head(10)

    fig = px.bar(
        feature_importance,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top 10 Features Influencing Churn Prediction",
        labels={"Importance": "Importance Score", "Feature": "Feature Name"},
        color="Importance",
        color_continuous_scale="Purples"
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    **Feature Importance Explanation:**
    - Feature importance is calculated using XGBoost Gain method
    - Higher values indicate features that contribute more to prediction accuracy
    - These features have the strongest influence on customer churn decisions
    """)
