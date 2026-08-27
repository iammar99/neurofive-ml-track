# ==========================================
#         Importing Libraries
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from imblearn.over_sampling import SMOTE


# ==========================================
#          Importing DataSet
# ==========================================

df = pd.read_csv(
    "G:\\Internships\\Neurofive ML\\Week 5\\Task 1\\creditcard.csv"
)

print("Dataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())


# ==========================================
#          Basic Information
# ==========================================

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum().sum())


# ==========================================
#          Check Class Balance
# ==========================================

print("\n==========================================")
print("             Class Balance")
print("==========================================")

print(df["Class"].value_counts())

print("\nClass Percentage:")

print(
    df["Class"].value_counts(normalize=True) * 100
)


# ==========================================
#       Visualize Class Balance
# ==========================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Class"
)

plt.title("Credit Card Fraud Class Distribution")

plt.xlabel("Class (0 = Normal, 1 = Fraud)")

plt.ylabel("Number of Transactions")

plt.show()


# ==========================================
#          Features and Target
# ==========================================

X = df.drop(
    "Class",
    axis=1
)

y = df["Class"]


print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print("Class")


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
#          Standard Scaling
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ==========================================
#       Model BEFORE SMOTE
# ==========================================

print("\n==========================================")
print("       Model Before SMOTE")
print("==========================================")


model_before = LogisticRegression(
    max_iter=1000
)

model_before.fit(
    X_train_scaled,
    y_train
)


# Predictions

y_pred_before = model_before.predict(
    X_test_scaled
)


# ==========================================
#      Evaluation BEFORE SMOTE
# ==========================================

accuracy_before = accuracy_score(
    y_test,
    y_pred_before
)

precision_before = precision_score(
    y_test,
    y_pred_before
)

recall_before = recall_score(
    y_test,
    y_pred_before
)

f1_before = f1_score(
    y_test,
    y_pred_before
)


print("Accuracy:", accuracy_before)
print("Precision:", precision_before)
print("Recall:", recall_before)
print("F1 Score:", f1_before)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred_before
    )
)


# ==========================================
#             Apply SMOTE
# ==========================================

print("\n==========================================")
print("              Applying SMOTE")
print("==========================================")


smote = SMOTE(
    random_state=42
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train_scaled,
    y_train
)


print("Before SMOTE:")

print(y_train.value_counts())


print("\nAfter SMOTE:")

print(y_train_smote.value_counts())


# ==========================================
#       Model AFTER SMOTE
# ==========================================

print("\n==========================================")
print("          Model After SMOTE")
print("==========================================")


model_after = LogisticRegression(
    max_iter=1000
)

model_after.fit(
    X_train_smote,
    y_train_smote
)


# Predictions

y_pred_after = model_after.predict(
    X_test_scaled
)


# ==========================================
#       Evaluation AFTER SMOTE
# ==========================================

accuracy_after = accuracy_score(
    y_test,
    y_pred_after
)

precision_after = precision_score(
    y_test,
    y_pred_after
)

recall_after = recall_score(
    y_test,
    y_pred_after
)

f1_after = f1_score(
    y_test,
    y_pred_after
)


print("Accuracy:", accuracy_after)
print("Precision:", precision_after)
print("Recall:", recall_after)
print("F1 Score:", f1_after)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred_after
    )
)


# ==========================================
#       Before vs After Comparison
# ==========================================

results = pd.DataFrame({

    "Model": [
        "Before SMOTE",
        "After SMOTE"
    ],

    "Accuracy": [
        accuracy_before,
        accuracy_after
    ],

    "Precision": [
        precision_before,
        precision_after
    ],

    "Recall": [
        recall_before,
        recall_after
    ],

    "F1 Score": [
        f1_before,
        f1_after
    ]
})


print("\n==========================================")
print("        Before vs After SMOTE")
print("==========================================")

print(results)


# ==========================================
#          Comparison Plot
# ==========================================

results_plot = results.set_index(
    "Model"
)

results_plot[
    ["Precision", "Recall", "F1 Score"]
].plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title(
    "Model Performance Before vs After SMOTE"
)

plt.ylabel("Score")

plt.ylim(0, 1)

plt.xticks(rotation=0)

plt.legend()

plt.show()


# ==========================================
#       Why Accuracy is Misleading
# ==========================================

print("\n==========================================")
print("       Why Accuracy is Misleading")
print("==========================================")

print("""
The Credit Card Fraud Detection dataset is highly
imbalanced because normal transactions are much more
common than fraudulent transactions.

If a model predicts almost every transaction as normal,
it can achieve very high accuracy while failing to detect
fraudulent transactions.

For example, if 99.8% of transactions are normal, a model
that predicts 'Normal' for every transaction could achieve
around 99.8% accuracy, but it would detect 0% of fraud.

Therefore, Precision, Recall and F1 Score are more useful
for evaluating fraud detection models.

Recall is especially important because it tells us how
many actual fraudulent transactions the model successfully
detects.
""")
