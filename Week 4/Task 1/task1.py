# ==========================================
#         Importing Libraries
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

import joblib


# ==========================================
#          Importing DataSet
# ==========================================

df = pd.read_csv(
    "G:\\Internships\\Neurofive ML\\Week 4\\Task 1\\Telco-Customer-Churn.csv"
)

print("Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())


# ==========================================
#           Basic Information
# ==========================================

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())


# ==========================================
#          Cleaning the Data
# ==========================================

# Convert TotalCharges to numeric

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)


# Remove customer ID because it is
# not useful for prediction

df = df.drop(
    "customerID",
    axis=1
)


# ==========================================
#       Feature Engineering
# ==========================================

# Feature 1:
# Create a new feature showing the average
# monthly charge over the customer's tenure

df["AverageCharge"] = (
    df["TotalCharges"] /
    df["tenure"].replace(0, np.nan)
)


# Feature 2:
# Create a simple feature showing whether
# the customer has been with the company
# for more than one year

df["LongTermCustomer"] = (
    df["tenure"] >= 12
).astype(int)


print("\nNew Features Created:")
print("1. AverageCharge")
print("2. LongTermCustomer")


# ==========================================
#        Create X and y
# ==========================================

X = df.drop(
    "Churn",
    axis=1
)

y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})


# ==========================================
#    Identify Numerical and Categorical
# ==========================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()


print("\nNumerical Features:")
print(numeric_features)

print("\nCategorical Features:")
print(categorical_features)


# ==========================================
#           Train/Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# ==========================================
#       ColumnTransformer
# ==========================================

# Numerical columns:
# Fill missing values + StandardScaler

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# Categorical columns:
# Fill missing values + OneHotEncoder

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                drop="first"
            )
        )
    ]
)


# Combine numerical and categorical processing

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ==========================================
#          Create Final Pipeline
# ==========================================

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            )
        )
    ]
)


# ==========================================
#            Train Pipeline
# ==========================================

print("\nTraining Pipeline...")

pipeline.fit(
    X_train,
    y_train
)

print("Pipeline Training Completed!")


# ==========================================
#             Predictions
# ==========================================

y_pred = pipeline.predict(
    X_test
)


# ==========================================
#            Model Evaluation
# ==========================================

print("\n==========================================")
print("          Pipeline Results")
print("==========================================")

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)


print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ==========================================
#       Feature Engineering Comparison
# ==========================================

print("\n==========================================")
print("      Engineered Features Check")
print("==========================================")

print("\nNew Features:")
print("- AverageCharge")
print("- LongTermCustomer")

print("\nPipeline Accuracy with Engineered Features:")
print(accuracy)


# ==========================================
#       Check Feature Importance
# ==========================================

# Logistic Regression uses coefficients
# instead of feature_importances_

print("\nModel used:")
print("Logistic Regression")

print("\nThe engineered features were included")
print("inside the pipeline and used during training.")


# ==========================================
#            Save Pipeline
# ==========================================

joblib.dump(
    pipeline,
    "G:\\Internships\\Neurofive ML\\Week 4\\Task 1\\telco_churn_pipeline.pkl"
)


print("\n==========================================")
print("        Pipeline Saved Successfully")
print("==========================================")

print(
    "Saved as: telco_churn_pipeline.pkl"
)


import os

print("Pipeline saved at:")
print(os.path.abspath("telco_churn_pipeline.pkl"))
