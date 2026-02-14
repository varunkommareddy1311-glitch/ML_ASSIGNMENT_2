import streamlit as st
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

st.title(" Heart Disease Prediction Dashboard")

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
    st.subheader("Dataset Preview")
    st.write(df.head())

    if "HeartDisease" in df.columns:
        X = df.drop("HeartDisease", axis=1)
        y = df["HeartDisease"]
    else:
        X = df
        y = None

    X = preprocess(X)
    X_scaled = scaler.transform(X)

    predictions = model.predict(X_scaled)

    st.subheader("Sample Predictions")
    st.write(predictions[:10])

    # ================= Metrics =================

    if y is not None:

        acc = accuracy_score(y, predictions)
        prec = precision_score(y, predictions, zero_division=0)
        rec = recall_score(y, predictions, zero_division=0)
        f1 = f1_score(y, predictions, zero_division=0)
        mcc = matthews_corrcoef(y, predictions)

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_scaled)[:,1]
            auc = roc_auc_score(y, probs)
        else:
            probs = None
            auc = None

        st.subheader("Evaluation Metrics")

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{acc:.3f}")
        col2.metric("Precision", f"{prec:.3f}")
        col3.metric("Recall", f"{rec:.3f}")

        col4, col5, col6 = st.columns(3)
        col4.metric("F1 Score", f"{f1:.3f}")
        col5.metric("MCC", f"{mcc:.3f}")
        if auc is not None:
            col6.metric("AUC", f"{auc:.3f}")

        # ================= Confusion Matrix =================

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y, predictions)

        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"],
            ax=ax_cm
        )

        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        st.pyplot(fig_cm)

        # ================= Classification Report =================

        st.subheader("Classification Report")

        report = classification_report(
            y,
            predictions,
            target_names=["No Disease", "Disease"],
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.3f}"))

        # ================= ROC Curve =================

        if probs is not None:
            st.subheader("ROC Curve")

            fpr, tpr, _ = roc_curve(y, probs)

            fig_roc, ax_roc = plt.subplots()
            ax_roc.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
            ax_roc.plot([0,1], [0,1], linestyle="--")
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.legend()
            st.pyplot(fig_roc)

# =========================
# Manual Prediction
# =========================

st.subheader("Manual Prediction")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 20, 100, 50)
    sex = st.selectbox("Sex", ["M","F"])
    chest = st.selectbox("Chest Pain Type", ["TA","ATA","NAP","ASY"])
    bp = st.number_input("Resting BP", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 600, 200)

with col2:
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

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(scaled)[0][1]
    else:
        prob = 0.0

    if pred == 1:
        st.error(f"High Risk (Probability: {prob:.2f})")
    else:
        st.success(f"Low Risk (Probability: {prob:.2f})")
