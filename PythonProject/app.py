from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import pickle
import pandas as pd
import plotly.express as px

# Set Streamlit layout to wide
st.set_page_config(layout="wide")

# Load the trained model
with open("templates/best_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load the MinMaxScaler
with open("templates/scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Define the input features for the model
feature_names = [
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
    "EstimatedSalary", "Geography_France", "Geography_Germany", "Geography_Spain",
    "Gender_Female", "Gender_Male", "HasCrCard_0", "HasCrCard_1",
    "IsActiveMember_0", "IsActiveMember_1"
]

# Columns requiring scaling
scale_vars = ["CreditScore", "EstimatedSalary", "Tenure", "Balance", "Age", "NumOfProducts"]

# Default values
default_values = {
    "CreditScore": 600,
    "Age": 30,
    "Tenure": 2,
    "Balance": 8000,
    "NumOfProducts": 2,
    "EstimatedSalary": 60000,
    "Geography_France": 1,
    "Geography_Germany": 0,
    "Geography_Spain": 0,
    "Gender_Female": 1,
    "Gender_Male": 0,
    "HasCrCard_0": 0,
    "HasCrCard_1": 1,
    "IsActiveMember_0": 0,
    "IsActiveMember_1": 1
}

# Sidebar setup
st.sidebar.image("templates/Pic 1.PNG", use_container_width=True)
st.sidebar.header("User Inputs")

# Collect user inputs
user_inputs = {}
for feature in feature_names:
    if feature in scale_vars:
        user_inputs[feature] = st.sidebar.number_input(feature, value=default_values[feature], step=1)
    else:
        user_inputs[feature] = st.sidebar.selectbox(feature, [0, 1], index=default_values[feature])

# Convert inputs to DataFrame
input_data = pd.DataFrame([user_inputs])

# Apply MinMaxScaler
input_data_scaled = input_data.copy()
input_data_scaled[scale_vars] = scaler.transform(input_data[scale_vars])

# App Header
st.image("templates/Pic 2.PNG", use_container_width=True)
st.title("Customer Churn Prediction")

# Page Layout
left_col, right_col = st.columns(2)

# Left Page: Feature Importance
with left_col:
    st.header("Feature Importance")
    try:
        feature_importance_df = pd.read_excel(
            "templates/feature_importance.xlsx",
            usecols=["Feature", "Feature Importance Score"],
            engine="openpyxl"
        )
        fig = px.bar(
            feature_importance_df.sort_values(by="Feature Importance Score", ascending=False),
            x="Feature Importance Score",
            y="Feature",
            orientation="h",
            title="Feature Importance",
            labels={"Feature Importance Score": "Importance", "Feature": "Features"},
            width=400,
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    except FileNotFoundError:
        st.info("⚠️ Feature importance file not found. Please add 'feature_importance.xlsx' in templates folder.")
    except ImportError:
        st.error("⚠️ 'openpyxl' is required to read Excel files. Install it using `pip install openpyxl`.")

# Right Page: Prediction
with right_col:
    st.header("Prediction")
    if st.button("Predict"):
        probabilities = model.predict_proba(input_data_scaled)[0]
        prediction = model.predict(input_data_scaled)[0]
        prediction_label = "Churned" if prediction == 1 else "Retain"

        st.subheader(f"Predicted Value: {prediction_label}")
        st.write(f"Predicted Probability: {probabilities[1]:.2%} (Churn)")
        st.write(f"Predicted Probability: {probabilities[0]:.2%} (Retain)")
        st.markdown(f"### Output: **{prediction_label}**")
