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

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

  from sklearn.metrics import (
      accuracy_score,
      precision_score,
      recall_score,
      f1_score,
      classification_report,
      confusion_matrix
  )

 
# ==========================================
#         Importing DataSet
# ==========================================


df = pd.read_csv("G:\\Internships\\Neurofive ML\\Week 3\\Task 1\\Telco-Customer-Churn.csv")

print("Shape:", df.shape)
df.head()



# ==========================================
#               Basic EDA
# ==========================================


print("Dataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())



print(df["Churn"].value_counts())

print("\nChurn Percentage:")
print(df["Churn"].value_counts(normalize=True) * 100)





# ==========================================
#     Check important numerical features
# ==========================================


df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df[["tenure", "MonthlyCharges", "TotalCharges"]].describe()


plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Churn",
    y="tenure"
)

plt.title("Churn vs Tenure")
plt.show()



plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Churn",
    y="MonthlyCharges"
)

plt.title("Churn vs Monthly Charges")
plt.show()




plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Contract",
    hue="Churn"
)

plt.title("Churn by Contract Type")
plt.xticks(rotation=15)
plt.show()



plt.figure(figsize=(10, 5))

sns.countplot(
    data=df,
    x="PaymentMethod",
    hue="Churn"
)

plt.title("Churn by Payment Method")
plt.xticks(rotation=30)
plt.show()






numeric_df = df[
    ["tenure", "MonthlyCharges", "TotalCharges"]
].copy()

numeric_df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

correlation = numeric_df.corr()

print(correlation)



plt.figure(figsize=(7, 5))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Matrix")
plt.show()


# ==========================================
#         Prepare the data
# ==========================================


df = df.drop("customerID", axis=1)
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)
X = df.drop("Churn", axis=1)
y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})


# ==========================================
# Identify numerical and categorical columns
# ==========================================


numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("Numerical Features:")
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

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))



# ==========================================
#            Preprocessing
# ==========================================

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            drop="first"
        ))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# ==========================================
#          Decision Tree Model
# ==========================================

decision_tree = Pipeline(
    steps=[
        ("preprocessor", preprocessor),

        ("classifier", DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced",
            max_depth=5
        ))
    ]
)


# ==========================================
#        Train Decision Tree
# ==========================================

decision_tree.fit(X_train, y_train)


# ==========================================
#       Decision Tree Predictions
# ==========================================

y_pred_tree = decision_tree.predict(X_test)


# ==========================================
#       Decision Tree Evaluation
# ==========================================

print("\n==========================================")
print("         Decision Tree Results")
print("==========================================")

print("Accuracy:",
      accuracy_score(y_test, y_pred_tree))

print("Precision:",
      precision_score(y_test, y_pred_tree))

print("Recall:",
      recall_score(y_test, y_pred_tree))

print("F1 Score:",
      f1_score(y_test, y_pred_tree))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_tree))


# ==========================================
#      Decision Tree Confusion Matrix
# ==========================================

cm_tree = confusion_matrix(
    y_test,
    y_pred_tree
)

print("\nConfusion Matrix:")
print(cm_tree)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm_tree,
    annot=True,
    fmt="d"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Decision Tree Confusion Matrix")

plt.show()


# ==========================================
#       Logistic Regression Model
# ==========================================

logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),

        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ]
)


# ==========================================
#     Train Logistic Regression
# ==========================================

logistic_model.fit(X_train, y_train)


# ==========================================
#    Logistic Regression Predictions
# ==========================================

y_pred_logistic = logistic_model.predict(X_test)


# ==========================================
#    Logistic Regression Evaluation
# ==========================================

print("\n==========================================")
print("       Logistic Regression Results")
print("==========================================")

print("Accuracy:",
      accuracy_score(y_test, y_pred_logistic))

print("Precision:",
      precision_score(y_test, y_pred_logistic))

print("Recall:",
      recall_score(y_test, y_pred_logistic))

print("F1 Score:",
      f1_score(y_test, y_pred_logistic))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred_logistic
))


# ==========================================
#           Model Comparison
# ==========================================

results = pd.DataFrame({

    "Model": [
        "Decision Tree",
        "Logistic Regression"
    ],

    "Accuracy": [
        accuracy_score(y_test, y_pred_tree),
        accuracy_score(y_test, y_pred_logistic)
    ],

    "Precision": [
        precision_score(y_test, y_pred_tree),
        precision_score(y_test, y_pred_logistic)
    ],

    "Recall": [
        recall_score(y_test, y_pred_tree),
        recall_score(y_test, y_pred_logistic)
    ],

    "F1 Score": [
        f1_score(y_test, y_pred_tree),
        f1_score(y_test, y_pred_logistic)
    ]
})

print("\n==========================================")
print("            Model Comparison")
print("==========================================")

print(results)


# ==========================================
#       Decision Tree Feature Importance
# ==========================================

tree_model = decision_tree.named_steps["classifier"]

feature_names = (
    decision_tree
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

importance = tree_model.feature_importances_

feature_importance = pd.DataFrame({

    "Feature": feature_names,

    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)


# ==========================================
#             Top 3 Features
# ==========================================

top_3_features = feature_importance.head(3)

print("\n==========================================")
print("      Top 3 Features Driving Churn")
print("==========================================")

print(top_3_features)


# ==========================================
#          Top 3 Features Plot
# ==========================================

plt.figure(figsize=(8, 5))

sns.barplot(
    data=top_3_features,
    x="Importance",
    y="Feature"
)

plt.title("Top 3 Features Driving Churn")

plt.show()
