# Neurofive ML Internship — Week 3, Task 1
## Model Evaluation & Hyperparameter Tuning (Titanic Classification)

### 📌 Overview
This task revisits the **Logistic Regression** Titanic survival model built in Task 1 and goes deeper into evaluation and optimization. It covers precision/recall/F1-score analysis, a discussion of why accuracy alone can be misleading on imbalanced data, hyperparameter tuning with `GridSearchCV`, and a before/after performance comparison.

---

### 🎯 Objectives
- Revisit the Titanic classification model from the previous task
- Calculate Precision, Recall, and F1-score using `classification_report`
- Explain why accuracy alone can be misleading for imbalanced datasets
- Tune at least 2 hyperparameters using `GridSearchCV`
- Compare the tuned model's performance to the original in a before/after table

---

### 🗂️ Dataset
`Titanic_Cleaned.csv` — the same cleaned Titanic dataset used in Task 1, with `Sex`, `Embarked`, `Ticket`, and `Cabin` one-hot encoded.

---

### ⚙️ Workflow

1. **Load & Prepare Data**
   Re-loaded the cleaned dataset, one-hot encoded categorical columns, and split into train/test sets (80/20, `random_state=42`) — identical setup to Task 1.

2. **Baseline Model**
   Trained the original `LogisticRegression(max_iter=1000)` model and evaluated it with:
   - `accuracy_score`
   - `confusion_matrix` (visualized as a heatmap)
   - `classification_report`
   - `precision_score`, `recall_score`, `f1_score`

3. **Hyperparameter Tuning**
   Used `GridSearchCV` to search over:
   - `C`: `[0.01, 0.1, 1, 10, 100]`
   - `solver`: `['liblinear', 'lbfgs']`

   with `cv=5` and `scoring='f1'` to find the best-performing combination.

4. **Tuned Model Evaluation**
   Extracted the best estimator (`grid_search.best_estimator_`), predicted on the test set, and recalculated accuracy, precision, recall, F1-score, and a new confusion matrix.

5. **Before/After Comparison**
   Built a simple comparison table (`pandas.DataFrame`) showing the original vs. tuned model side by side across all four metrics.

---

### 📊 Results

**Original Model — Confusion Matrix**
```
[[90 15]
 [19 55]]
```

**Original Model — Metrics**

| Metric | Value |
|--------|-------|
| Accuracy  | *0.8101* |
| Precision | *0.7857* |
| Recall    | *0.7432* |
| F1-score  | *0.7639* |

**Best Hyperparameters Found**

| Parameter | Value |
|-----------|-------|
| `C`       | *100* |
| `solver`  | *liblinear* |


**Before vs. After Comparison**

| Metric | Original Model | Tuned Model |
|--------|:---------------:|:------------:|
| Accuracy  | *81.01%*  | *83.8%* |
| Precision | *78.57% * | *83.58%* |
| Recall    | *74.32%* | *75.68%* |
| F1-score  | *76.39* | *79.43%* |

> ⚠️ Fill in the placeholders above with the exact numbers printed by your script (`comparison` DataFrame and `grid_search.best_params_`).

---

### ⚠️ Why Accuracy Alone Can Be Misleading (Plain English)

Accuracy just tells us the percentage of total predictions the model got right — but it doesn't say **what kind** of mistakes it's making, or **who** it's failing. On an imbalanced dataset (say, if 90% of passengers didn't survive), a model could simply predict "did not survive" every single time and still score 90% accuracy, despite being completely useless at identifying survivors. That's why we also look at **precision** (of everyone we predicted as survivors, how many actually survived?), **recall** (of everyone who actually survived, how many did we correctly catch?), and **F1-score** (a balance between the two). These metrics reveal how well the model handles the *minority class* — the group that's harder to predict and often the one we care about most — which raw accuracy can hide.

---

### 🔧 Note on the Script
The provided script uses `GridSearchCV` but does not import it. Make sure to add the following import near the top of the file, alongside the other `sklearn` imports:
```python
from sklearn.model_selection import GridSearchCV
```

---


### 📁 Suggested Project Structure
```
Week 3/Task 1/
│
├── Titanic_Cleaned.csv
├── task3.py
└── README.md
```

---

### ✅ Summary
This task extends the Titanic classification pipeline with deeper evaluation (precision, recall, F1-score) and systematic hyperparameter tuning via `GridSearchCV`, then quantifies the improvement — if any — over the original baseline model in a clear before/after comparison.