# Heart Disease Prediction using Machine Learning

## a. Problem Statement

Heart disease is one of the leading causes of mortality worldwide. Early prediction can help in timely intervention and prevention.  
The objective of this project is to build and evaluate **machine learning classification models** that predict whether a patient has heart disease based on clinical features such as age, sex, cholesterol, blood pressure, and ECG results.  

---

## b. Dataset Description

The dataset used is a **Heart Disease dataset** containing 918 patient records and 12 features.  

**Features include:**

| Feature | Description |
|---------|-------------|
| Age | Patient age in years |
| Sex | Patient gender (M/F) |
| ChestPainType | Type of chest pain (TA, ATA, NAP, ASY) |
| RestingBP | Resting blood pressure (mm Hg) |
| Cholesterol | Serum cholesterol (mg/dl) |
| FastingBS | Fasting blood sugar > 120 mg/dl (1 = true; 0 = false) |
| RestingECG | Resting electrocardiogram results (Normal, ST, LVH) |
| MaxHR | Maximum heart rate achieved |
| ExerciseAngina | Exercise-induced angina (Y/N) |
| Oldpeak | ST depression induced by exercise relative to rest |
| ST_Slope | Slope of peak exercise ST segment (Up, Flat, Down) |
| HeartDisease | Target variable: 1 = presence of heart disease, 0 = absence |

**Source:** Public dataset (Kaggle / UCI)  

Number of instances: 918  
Number of features: 12  

---

## c. Models Used

The following machine learning classification models were implemented on the dataset:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbor Classifier  
4. Naive Bayes Classifier (Gaussian)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)  

Each model was evaluated using the following metrics:

- Accuracy  
- AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)  

---

## d. Model Performance Comparison

| Model                | Accuracy  | AUC      | Precision | Recall   | F1 Score | MCC      |
|----------------------|----------|----------|-----------|---------|----------|----------|
| Logistic Regression  | 0.847826 | 0.900837 | 0.907216  | 0.822430 | 0.862745 | 0.697136 |
| Decision Tree        | 0.798913 | 0.803435 | 0.864583  | 0.775701 | 0.817734 | 0.599316 |
| KNN                  | 0.847826 | 0.922260 | 0.907216  | 0.822430 | 0.862745 | 0.697136 |
| Naive Bayes          | 0.842391 | 0.908970 | 0.882353  | 0.841121 | 0.861244 | 0.680137 |
| Random Forest        | 0.869565 | 0.939799 | 0.902913  | 0.869159 | 0.885714 | 0.734666 |
| XGBoost              | 0.869565 | 0.936643 | 0.919192  | 0.850467 | 0.883495 | 0.738723 |

---

## e. Observations on Model Performance

| Model                | Observation |
|----------------------|-------------|
| Logistic Regression  | Performs well with good precision and recall, suitable for linear relationships. |
| Decision Tree        | Slightly lower accuracy; may overfit on training data but interpretable. |
| KNN                  | Performs similar to Logistic Regression; sensitive to feature scaling. |
| Naive Bayes          | Slightly lower accuracy than KNN and Logistic Regression; performs well on categorical features. |
| Random Forest        | Best performance among models; robust and handles feature interactions well. |
| XGBoost              | Comparable to Random Forest; slightly better MCC; handles complex patterns efficiently. |

**Conclusion:** Ensemble models (Random Forest and XGBoost) achieved the best overall performance on this dataset, showing higher accuracy, AUC, and MCC scores compared to individual classifiers.

---
