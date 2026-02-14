import streamlit as st
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score, confusion_matrix
)

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

st.title("Heart Disease Prediction")

# =========================
# Load Models & ScaIer
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

model_name = st.selectbox("Choose Model", list(models.keys()))
model = models[model_name]

# =========================
# Upload CSV
# =========================

uploaded_file = st.file_uploader("Upload CSV Test Data", type=["csv"])

# =========================
# PREPROCESS FUNCTION
# =========================

def preprocess(df):
    df = df.copy()

    # encode categorical columns (same as training)
    df["Sex"] = df["Sex"].map({"M": 1, "F": 0})
    df["ChestPainType"] = df["ChestPainType"].map({"TA":0,"ATA":1,"NAP":2,"ASY":3})
    df["RestingECG"] = df["RestingECG"].map({"Normal":0,"ST":1,"LVH":2})
    df["ExerciseAngina"] = df["ExerciseAngina"].map({"N":0,"Y":1})
    df["ST_Slope"] = df["ST_Slope"].map({"Up":0,"Flat":1,"Down":2})

    df = df.fillna(0)

    return df

# =========================
# If CSV uploaded
# =========================

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview", df.head())

    if "HeartDisease" in df.columns:
        X = df.drop("HeartDisease", axis=1)
        y = df["HeartDisease"]
    else:
        X = df
        y = None

    X = preprocess(X)
    X_scaled = scaler.transform(X)

    predictions = model.predict(X_scaled)

    st.subheader("Predictions")
    st.write(predictions[:10])

    # ================= Metrics =================

    if y is not None:
        acc = accuracy_score(y, predictions)
        prec = precision_score(y, predictions)
        rec = recall_score(y, predictions)
        f1 = f1_score(y, predictions)
        mcc = matthews_corrcoef(y, predictions)

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_scaled)[:,1]
            auc = roc_auc_score(y, probs)
        else:
            auc = None

        st.subheader("Evaluation Metrics")
        st.write(f"Accuracy: {acc:.3f}")
        st.write(f"Precision: {prec:.3f}")
        st.write(f"Recall: {rec:.3f}")
        st.write(f"F1 Score: {f1:.3f}")
        st.write(f"MCC: {mcc:.3f}")

        if auc:
            st.write(f"AUC: {auc:.3f}")

        # Confusion matrix
        cm = confusion_matrix(y, predictions)
        fig, ax = plt.subplots()
        ax.matshow(cm)
        for i in range(len(cm)):
            for j in range(len(cm)):
                ax.text(j, i, cm[i, j], ha='center', va='center')

        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.subheader("Confusion Matrix")
        st.pyplot(fig)

# =========================
# Manual Prediction
# =========================

st.subheader("Manual Prediction")

age = st.number_input("Age", 20, 100, 50)
sex = st.selectbox("Sex", ["M","F"])
chest = st.selectbox("Chest Pain Type", ["TA","ATA","NAP","ASY"])
bp = st.number_input("Resting BP", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 600, 200)
fbs = st.selectbox("Fasting BS > 120", [0,1])
ecg = st.selectbox("Resting ECG", ["Normal","ST","LVH"])
hr = st.number_input("Max HR", 60, 220, 150)
angina = st.selectbox("Exercise Angina", ["Y","N"])
oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0)
slope = st.selectbox("ST Slope", ["Up","Flat","Down"])

if st.button("Predict"):

    input_df = pd.DataFrame([{
        "Age":age, "Sex":sex, "ChestPainType":chest,
        "RestingBP":bp, "Cholesterol":chol,
        "FastingBS":fbs, "RestingECG":ecg,
        "MaxHR":hr, "ExerciseAngina":angina,
        "Oldpeak":oldpeak, "ST_Slope":slope
    }])

    input_df = preprocess(input_df)
    scaled = scaler.transform(input_df)

    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1]

    if pred == 1:
        st.error(f"High Risk (probability {prob:.2f})")
    else:
        st.success(f"Low Risk (probability {prob:.2f})")
