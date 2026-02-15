import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Load dataset
#df = pd.read_csv("D:\BITS Mtech\ML\ML_Assignment_2\data\dataset.csv")
df = pd.read_csv("D:\\BITS Mtech\\ML\\ML_Assignment_2\\data\\dataset.csv")


# Encode categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# Split features and target
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Scale features (important for KNN & Logistic)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model_dir = r"D:\BITS Mtech\ML\ML_Assignment_2\model"
os.makedirs(model_dir, exist_ok=True)

# Models dictionary
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(eval_metric='logloss')
}

results = []

print("\nModel Performance:\n")

for name, model in models.items():
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

    results.append(metrics)

    # Save model
    #joblib.dump(model, f"model/{name.replace(' ', '_').lower()}.pkl")
    joblib.dump(model, os.path.join(model_dir, f"{name.replace(' ', '_').lower()}.pkl"))


# Save scaler
#joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))


# Print results
results_df = pd.DataFrame(results)
print(results_df)

# Save results
results_df.to_csv("model_results.csv", index=False)

print("\n Models trained & saved successfully!")
