# ==========================================
#         Importing Libraries
# ==========================================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)
 
# ==========================================
#         Importing DataSet
# ==========================================


data = pd.read_csv("G:\\Internships\\Neurofive ML\\Week 2\\Task 1\\Titanic_Cleaned.csv")




# ==========================================
#     Encoding categorical columns 
# ==========================================


# Encode categorical columns (e.g., "Sex", "Embarked") using OneHotEncoder or pd.get_dummies()

df_encoded = pd.get_dummies(data, columns=['Sex', 'Embarked','Ticket','Cabin'], drop_first=True)


# ==========================================
#           Splitting dataset
# ==========================================


# Use scikit-learn to split your cleaned dataset into training and test sets using train_test_split

X = df_encoded.drop(columns=['Survived','Name']) 
y = df_encoded['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training features shape: {X_train.shape}")
print(f"Testing features shape: {X_test.shape}")





# ==========================================
#           Selecting a Model
# ==========================================


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)





# ==========================================
#    Testing Accuracy & Confusion Matrix
# ==========================================



y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")



# Model Accuracy: 81.01%


cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)


# ==========================================
#           Visualization
# ==========================================

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Did Not Survive', 'Survived'], 
            yticklabels=['Did Not Survive', 'Survived'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()







# ==========================================
# Classification Report - Original Model
# ==========================================

print("\nOriginal Model Classification Report:")
print(classification_report(y_test, y_pred))




# ==========================================
# Saving Original Model Metrics
# ==========================================

original_precision = precision_score(
    y_test,
    y_pred
)

original_recall = recall_score(
    y_test,
    y_pred
)

original_f1 = f1_score(
    y_test,
    y_pred
)

print(f"Precision: {original_precision:.4f}")
print(f"Recall:    {original_recall:.4f}")
print(f"F1-score:  {original_f1:.4f}")



# ==========================================
# Hyperparameter Tuning using GridSearchCV
# ==========================================

param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'lbfgs']
}

grid_search = GridSearchCV(
    estimator=LogisticRegression(max_iter=2000),
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=1
)

grid_search.fit(X_train, y_train)



# ==========================================
# Best Hyperparameters
# ==========================================

print("Best Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation F1-score:")
print(grid_search.best_score_)


# ==========================================
# Get Best Tuned Model
# ==========================================

tuned_model = grid_search.best_estimator_

print("Best Tuned Model:")
print(tuned_model)



# ==========================================
# Test Tuned Model
# ==========================================

y_pred_tuned = tuned_model.predict(X_test)

print(classification_report(y_test, y_pred_tuned))






# ==========================================
# Saving Tuned Model Metrics
# ==========================================

tuned_accuracy = accuracy_score(
    y_test,
    y_pred_tuned
)

tuned_precision = precision_score(
    y_test,
    y_pred_tuned
)

tuned_recall = recall_score(
    y_test,
    y_pred_tuned
)

tuned_f1 = f1_score(
    y_test,
    y_pred_tuned
)

print("\nTuned Model Metrics:")
print(f"Accuracy:  {tuned_accuracy:.4f}")
print(f"Precision: {tuned_precision:.4f}")
print(f"Recall:    {tuned_recall:.4f}")
print(f"F1-score:  {tuned_f1:.4f}")







# ==========================================
# Before vs After Comparison
# ==========================================

comparison = pd.DataFrame({
    'Metric': [
        'Accuracy',
        'Precision',
        'Recall',
        'F1-score'
    ],
    
    'Original Model': [
        accuracy,
        original_precision,
        original_recall,
        original_f1
    ],
    
    'Tuned Model': [
        tuned_accuracy,
        tuned_precision,
        tuned_recall,
        tuned_f1
    ]
})

print("\n==========================================")
print("BEFORE vs AFTER COMPARISON")
print("==========================================")

print(comparison)



# ==========================================
# Tuned Model Confusion Matrix
# ==========================================

cm_tuned = confusion_matrix(
    y_test,
    y_pred_tuned
)

print("\nTuned Model Confusion Matrix:")
print(cm_tuned)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm_tuned,
    annot=True,
    fmt='d',
    cmap='Greens',
    xticklabels=['Did Not Survive', 'Survived'],
    yticklabels=['Did Not Survive', 'Survived']
)

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Tuned Logistic Regression - Confusion Matrix')

plt.show()
