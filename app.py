"""
import streamlit as st
import pandas as pd
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.title("Heart Disease Prediction App")

st.write("Upload test dataset and evaluate ML models")

# Load scaler
#scaler = joblib.load("models/scaler.pkl")
scaler = joblib.load(r"D:\BITS Mtech\ML\ML_Assignment_2\model\scaler.pkl")

# Model paths
models = {
    "Logistic Regression": "models/logistic_regression.pkl",
    "Decision Tree": "models/decision_tree.pkl",
    "KNN": "models/knn.pkl",
    "Naive Bayes": "models/naive_bayes.pkl",
    "Random Forest": "models/random_forest.pkl",
    "XGBoost": "models/xgboost.pkl"
}

# Upload dataset
uploaded_file = st.file_uploader("Upload Test CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview", df.head())

    if "HeartDisease" not in df.columns:
        st.error("Target column 'HeartDisease' not found!")
    else:
        X = df.drop("HeartDisease", axis=1)
        y = df["HeartDisease"]

        # Scale
        X_scaled = scaler.transform(X)

        # Model selection
        model_name = st.selectbox("Select Model", list(models.keys()))

        if st.button("Evaluate Model"):
            model = joblib.load(models[model_name])

            y_pred = model.predict(X_scaled)
            y_prob = model.predict_proba(X_scaled)[:, 1]

            # Metrics
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred)
            recall = recall_score(y, y_pred)
            f1 = f1_score(y, y_pred)
            auc = roc_auc_score(y, y_prob)
            mcc = matthews_corrcoef(y, y_pred)

            st.subheader("Evaluation Metrics")

            col1, col2, col3 = st.columns(3)

            col1.metric("Accuracy", f"{accuracy:.2f}")
            col2.metric("Precision", f"{precision:.2f}")
            col3.metric("Recall", f"{recall:.2f}")

            col1.metric("F1 Score", f"{f1:.2f}")
            col2.metric("AUC", f"{auc:.2f}")
            col3.metric("MCC", f"{mcc:.2f}")

            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y, y_pred)
            st.write(cm)

            st.subheader("Classification Report")
            st.text(classification_report(y, y_pred))
            """

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

st.title(" Heart Disease Prediction App")

# =========================
# Load Models & Scaler
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

models = {
    "Logistic Regression": joblib.load(os.path.join(MODEL_DIR, "logistic_regression.pkl")),
    "Decision Tree": joblib.load(os.path.join(MODEL_DIR, "decision_tree.pkl")),
    "KNN": joblib.load(os.path.join(MODEL_DIR, "knn.pkl")),
    "Naive Bayes": joblib.load(os.path.join(MODEL_DIR, "naive_bayes.pkl")),
    "Random Forest": joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl")),
    "XGBoost": joblib.load(os.path.join(MODEL_DIR, "xgboost.pkl")),
}

# =========================
# Sidebar Model Selection
# =========================

model_name = st.sidebar.selectbox("Select Model", list(models.keys()))
model = models[model_name]

st.sidebar.markdown("---")
st.sidebar.write("Upload CSV (optional test data)")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# =========================
# User Inputs
# =========================

st.subheader("Enter Patient Details")

age = st.number_input("Age", 20, 100, 50)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["TA", "ATA", "NAP", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure", 80, 200, 120)
cholesterol = st.number_input("Cholesterol", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.number_input("Max Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise Angina", ["Y", "N"])
oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# =========================
# Create DataFrame
# =========================

input_data = {
    "Age": age,
    "Sex": sex,
    "ChestPainType": chest_pain,
    "RestingBP": resting_bp,
    "Cholesterol": cholesterol,
    "FastingBS": fasting_bs,
    "RestingECG": resting_ecg,
    "MaxHR": max_hr,
    "ExerciseAngina": exercise_angina,
    "Oldpeak": oldpeak,
    "ST_Slope": st_slope,
}

X = pd.DataFrame([input_data])

# =========================
# Encode categorical values
# (MUST match training)
# =========================

X["Sex"] = X["Sex"].map({"M": 1, "F": 0})

X["ChestPainType"] = X["ChestPainType"].map({
    "TA": 0, "ATA": 1, "NAP": 2, "ASY": 3
})

X["RestingECG"] = X["RestingECG"].map({
    "Normal": 0, "ST": 1, "LVH": 2
})

X["ExerciseAngina"] = X["ExerciseAngina"].map({
    "N": 0, "Y": 1
})

X["ST_Slope"] = X["ST_Slope"].map({
    "Up": 0, "Flat": 1, "Down": 2
})

# =========================
# Prediction
# =========================

if st.button("Predict"):

    X_scaled = scaler.transform(X)

    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠ High Risk of Heart Disease")
    else:
        st.success("Low Risk of Heart Disease")

    st.write(f"**Probability:** {probability:.2f}")

# =========================
# CSV Upload Prediction
# =========================

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("Uploaded Data")
    st.dataframe(df.head())

    # encode same way
    df["Sex"] = df["Sex"].map({"M": 1, "F": 0})
    df["ChestPainType"] = df["ChestPainType"].map({"TA":0,"ATA":1,"NAP":2,"ASY":3})
    df["RestingECG"] = df["RestingECG"].map({"Normal":0,"ST":1,"LVH":2})
    df["ExerciseAngina"] = df["ExerciseAngina"].map({"N":0,"Y":1})
    df["ST_Slope"] = df["ST_Slope"].map({"Up":0,"Flat":1,"Down":2})

    df_scaled = scaler.transform(df)

    preds = model.predict(df_scaled)

    df["Prediction"] = preds

    st.subheader("Predictions")
    st.dataframe(df)
