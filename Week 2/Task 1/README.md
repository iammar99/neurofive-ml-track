# Neurofive ML Internship — Week 2, Task 1

## Titanic Survival Prediction using Logistic Regression

### 📌 Overview
This task builds a **Logistic Regression** model to predict passenger survival on the Titanic using the cleaned Titanic dataset. It covers encoding categorical features, splitting data into train/test sets, training the model, and evaluating performance with an accuracy score and confusion matrix.

---

### 🎯 Objectives
- Split the cleaned dataset into training and test sets using `train_test_split`
- Train a Logistic Regression model to predict survival
- Encode categorical columns (`Sex`, `Embarked`) using `pd.get_dummies()`
- Evaluate the model using `accuracy_score`
- Generate and interpret a confusion matrix

---

### 🗂️ Dataset
`Titanic_Cleaned.csv` — a pre-cleaned version of the classic Titanic dataset, containing passenger details such as `Sex`, `Embarked`, `Ticket`, `Cabin`, `Name`, and the target column `Survived`.

---

### ⚙️ Workflow

1. **Import Libraries**
   `pandas`, `numpy`, `seaborn`, `matplotlib`, and relevant modules from `sklearn`.

2. **Load Dataset**
   Read the cleaned CSV file into a DataFrame.

3. **Encode Categorical Columns**
   Used `pd.get_dummies()` on `Sex`, `Embarked`, `Ticket`, and `Cabin` (with `drop_first=True` to avoid multicollinearity).

4. **Feature/Target Split**
   - `X` = all columns except `Survived` and `Name`
   - `y` = `Survived`

5. **Train/Test Split**
   Split into 80% training and 20% testing data using `train_test_split(random_state=42)`.

6. **Model Training**
   Trained a `LogisticRegression(max_iter=1000)` model on the training set.

7. **Evaluation**
   - Predicted on the test set
   - Calculated accuracy using `accuracy_score`
   - Generated a confusion matrix using `confusion_matrix`
   - Visualized the confusion matrix as a heatmap with `seaborn`

---

### 📊 Results

**Model Accuracy:** `81.01%`

**Confusion Matrix:**

|                     | Predicted: Did Not Survive | Predicted: Survived |
|---------------------|:---------------------------:|:--------------------:|
| **Actual: Did Not Survive** | 90 (TN) | 15 (FP) |
| **Actual: Survived**        | 19 (FN) | 55 (TP) |

```
[[90 15]
 [19 55]]
```

---

### 🔍 Confusion Matrix — Interpretation

- **True Negatives (90):** Passengers who did not survive, and the model correctly predicted they did not survive.
- **False Positives (15):** Passengers who did not survive, but the model incorrectly predicted they survived.
- **False Negatives (19):** Passengers who survived, but the model incorrectly predicted they did not survive.
- **True Positives (55):** Passengers who survived, and the model correctly predicted they survived.

**What this tells us:**
- The model is fairly good overall, correctly classifying **145 out of 179** test passengers (**81.01% accuracy**).
- It performs slightly better at identifying passengers who **did not survive** (90 correct out of 105, ~85.7%) than those who **did survive** (55 correct out of 74, ~74.3%).
- The **19 false negatives** are the more "costly" errors here — survivors the model missed — suggesting the model leans slightly conservative in predicting survival, possibly due to class imbalance or feature encoding of high-cardinality columns like `Ticket` and `Cabin`.
- Precision for the "Survived" class is **55/(55+15) ≈ 78.6%**, and recall is **55/(55+19) ≈ 74.3%**, indicating a reasonable balance between false alarms and missed survivors.

---



### ✅ Summary
This task demonstrates a complete, simple ML pipeline — from encoding categorical data and splitting the dataset, to training a Logistic Regression classifier and evaluating it with accuracy and a confusion matrix — achieving **81.01% accuracy** in predicting Titanic passenger survival.