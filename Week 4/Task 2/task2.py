# ==========================================
#         Importing Libraries
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from xgboost import XGBClassifier


# ==========================================
#          Importing DataSet
# ==========================================

df = pd.read_csv(
    "G:\\Internships\\Neurofive ML\\Week 3\\Task 1\\Telco-Customer-Churn.csv"
)

print("Shape:", df.shape)


# ==========================================
#          Cleaning the Data
# ==========================================

# Convert TotalCharges into numeric

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)


# Remove customer ID

df = df.drop(
    "customerID",
    axis=1
)


# ==========================================
#          Feature Engineering
# ==========================================

# Feature 1:
# Average monthly charge during tenure

df["AverageCharge"] = (
    df["TotalCharges"] /
    df["tenure"].replace(0, np.nan)
)


# Feature 2:
# Whether customer stayed for at least 1 year

df["LongTermCustomer"] = (
    df["tenure"] >= 12
).astype(int)


print("\nNew Features:")
print("AverageCharge")
print("LongTermCustomer")


# ==========================================
#          Features and Target
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
#     Numerical and Categorical Columns
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
#          Train/Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
#          Preprocessing
# ==========================================

# Numerical columns

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


# Categorical columns

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


# Combine preprocessing

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
#          Logistic Regression
#          Earlier Model
# ==========================================

logistic_model = Pipeline(
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


# Train Logistic Regression

logistic_model.fit(
    X_train,
    y_train
)


# Predictions

y_pred_logistic = logistic_model.predict(
    X_test
)


# ==========================================
#          Random Forest
# ==========================================

random_forest = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight="balanced"
            )
        )
    ]
)


# Train Random Forest

random_forest.fit(
    X_train,
    y_train
)


# Predictions

y_pred_rf = random_forest.predict(
    X_test
)


# ==========================================
#             XGBoost
# ==========================================

xgb_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
                eval_metric="logloss"
            )
        )
    ]
)


# Train XGBoost

xgb_model.fit(
    X_train,
    y_train
)


# Predictions

y_pred_xgb = xgb_model.predict(
    X_test
)


# ==========================================
#          Model Evaluation
# ==========================================

def evaluate_model(
    model_name,
    y_test,
    predictions
):

    print("\n==========================================")
    print(model_name)
    print("==========================================")

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions
    )

    recall = recall_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions
        )
    )

    return accuracy, precision, recall, f1


# Evaluate all three models

logistic_results = evaluate_model(
    "Logistic Regression",
    y_test,
    y_pred_logistic
)

rf_results = evaluate_model(
    "Random Forest",
    y_test,
    y_pred_rf
)

xgb_results = evaluate_model(
    "XGBoost",
    y_test,
    y_pred_xgb
)


# ==========================================
#          Model Comparison
# ==========================================

results = pd.DataFrame({

    "Model": [
        "Logistic Regression",
        "Random Forest",
        "XGBoost"
    ],

    "Accuracy": [
        logistic_results[0],
        rf_results[0],
        xgb_results[0]
    ],

    "Precision": [
        logistic_results[1],
        rf_results[1],
        xgb_results[1]
    ],

    "Recall": [
        logistic_results[2],
        rf_results[2],
        xgb_results[2]
    ],

    "F1 Score": [
        logistic_results[3],
        rf_results[3],
        xgb_results[3]
    ]
})


print("\n==========================================")
print("             Model Comparison")
print("==========================================")

print(results)


# ==========================================
#       Get Feature Names
# ==========================================

feature_names = (
    random_forest
    .named_steps["preprocessor"]
    .get_feature_names_out()
)


# ==========================================
#       Random Forest Feature Importance
# ==========================================

rf_model = (
    random_forest
    .named_steps["model"]
)

rf_importance = rf_model.feature_importances_


rf_features = pd.DataFrame({

    "Feature": feature_names,

    "Importance": rf_importance
})


rf_features = rf_features.sort_values(
    by="Importance",
    ascending=False
)


print("\n==========================================")
print("     Random Forest Top 10 Features")
print("==========================================")

print(
    rf_features.head(10)
)


# ==========================================
#       XGBoost Feature Importance
# ==========================================

xgb_model_only = (
    xgb_model
    .named_steps["model"]
)

xgb_importance = (
    xgb_model_only.feature_importances_
)


xgb_features = pd.DataFrame({

    "Feature": feature_names,

    "Importance": xgb_importance
})


xgb_features = xgb_features.sort_values(
    by="Importance",
    ascending=False
)


print("\n==========================================")
print("          XGBoost Top 10 Features")
print("==========================================")

print(
    xgb_features.head(10)
)


# ==========================================
#       Random Forest Plot
# ==========================================

plt.figure(figsize=(8, 6))

sns.barplot(
    data=rf_features.head(10),
    x="Importance",
    y="Feature"
)

plt.title(
    "Random Forest - Top 10 Features"
)

plt.show()


# ==========================================
#          XGBoost Plot
# ==========================================

plt.figure(figsize=(8, 6))

sns.barplot(
    data=xgb_features.head(10),
    x="Importance",
    y="Feature"
)

plt.title(
    "XGBoost - Top 10 Features"
)

plt.show()


# ==========================================
#       Compare Important Features
# ==========================================

print("\n==========================================")
print("     Random Forest vs XGBoost")
print("==========================================")

print("\nRandom Forest Top 5:")
print(
    rf_features.head(5)["Feature"].tolist()
)

print("\nXGBoost Top 5:")
print(
    xgb_features.head(5)["Feature"].tolist()
)
