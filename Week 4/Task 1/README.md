# Neurofive ML Internship — Week 4, Task 1
## End-to-End Pipeline with Feature Engineering (Telco Customer Churn)

### 📌 Overview
This task rebuilds the Telco Customer Churn model as a **single, self-contained `sklearn` Pipeline** — combining a `ColumnTransformer` (scaling + encoding) and a Logistic Regression model into one object — then extends it with two new engineered features and saves the final trained pipeline to disk with `joblib`.

---

### 🎯 Objectives
- Reuse the Telco Customer Churn dataset (previously used in Week 3)
- Build a single `Pipeline` using `ColumnTransformer` — `StandardScaler` for numerical columns, `OneHotEncoder` for categorical columns
- Chain preprocessing and the model into one pipeline object
- Fit and evaluate the pipeline, confirming it matches or beats the earlier manual approach
- Engineer at least 2 new features and test whether they improve performance
- Save the final pipeline with `joblib`

---

### 🗂️ Dataset
`Telco-Customer-Churn.csv` — **7,043 rows × 21 columns**, no missing values, no duplicates. Target column `Churn` (`Yes`/`No`), mapped to `1`/`0`.

---

### ⚙️ Workflow

1. **Import Libraries**
   `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn` (pipeline, compose, preprocessing, impute, linear_model, metrics), and `joblib` for persistence.

2. **Load & Inspect Dataset**
   Confirmed shape (7,043 × 21), column dtypes via `.info()`, and zero missing values.

3. **Clean the Data**
   - Converted `TotalCharges` from `object` to numeric (`errors="coerce"`, producing some `NaN`s that are handled later by the pipeline's imputer)
   - Dropped `customerID` (non-predictive identifier)

4. **Feature Engineering**
   Two new features were created directly on the DataFrame before the train/test split:
   - **`AverageCharge`** = `TotalCharges / tenure` (with `tenure == 0` replaced by `NaN` to avoid division-by-zero) — captures the customer's typical monthly spend across their whole relationship with the company, which can differ from the current `MonthlyCharges` if pricing changed over time.
   - **`LongTermCustomer`** = binary flag, `1` if `tenure >= 12` months else `0` — a simple, interpretable signal for "has this customer passed the one-year mark," since the Week 3 EDA showed churn drops sharply as tenure increases.

5. **Define X / y**
   `X` = all columns except `Churn`; `y` = `Churn` mapped to `0`/`1`.

6. **Identify Feature Types**
   - **Numeric (6):** `SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`, `AverageCharge`, `LongTermCustomer`
   - **Categorical (15):** `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`

7. **Train/Test Split**
   80/20 split, `stratify=y`, `random_state=42` → **5,634 training samples**, **1,409 testing samples** (identical split configuration to Week 3, for a fair comparison).

8. **Single Pipeline with ColumnTransformer**
   - **Numeric branch:** median imputation → `StandardScaler`
   - **Categorical branch:** most-frequent imputation → `OneHotEncoder(drop="first", handle_unknown="ignore")`
   - Combined into one `ColumnTransformer`, then chained with `LogisticRegression(max_iter=1000, class_weight="balanced")` inside a single top-level `Pipeline([("preprocessor", ...), ("model", ...)])` — preprocessing and modeling fit together in one `.fit()` call, eliminating any risk of data leakage or train/test mismatch.

9. **Train & Predict**
   Fit the pipeline on `X_train`/`y_train`, predicted on `X_test`.

10. **Evaluate**
    Computed accuracy, precision, recall, F1-score, and a full classification report.

11. **Save the Pipeline**
    Persisted the entire fitted pipeline (preprocessing + model together) to `telco_churn_pipeline.pkl` using `joblib.dump()`, so it can be reloaded later with `joblib.load()` and used to predict on new raw data directly — no separate preprocessing step required.

---

### 📊 Results — Pipeline with Engineered Features

| Metric | Value |
|--------|:-----:|
| Accuracy | 73.60% |
| Precision | 50.17% |
| Recall | 78.88% |
| F1 Score | 61.33% |

Classification report:

| Class | Precision | Recall | F1-score | Support |
|-------|:---------:|:------:|:--------:|:-------:|
| 0 (No churn) | 0.90 | 0.72 | 0.80 | 1035 |
| 1 (Churn) | 0.50 | 0.79 | 0.61 | 374 |

---

### 🔁 Pipeline vs. Manual Approach (Week 3 Logistic Regression)

| Metric | Manual Approach (Week 3) | Pipeline + Engineered Features (Week 4) | Change |
|--------|:-------------------------:|:-----------------------------------------:|:------:|
| Accuracy | 73.81% | 73.60% | -0.21 pts |
| Precision | 50.43% | 50.17% | -0.26 pts |
| Recall | 78.34% | 78.88% | +0.54 pts |
| F1 Score | 61.36% | 61.33% | -0.03 pts |

**Confirming the pipeline works correctly:** the single-`Pipeline` approach reproduces essentially the **same performance** as the earlier manual `ColumnTransformer` + model setup from Week 3 (differences of a fraction of a percentage point across all metrics), confirming the pipeline is correctly wired and behaves identically to the manual preprocessing + modeling steps, just packaged into one reusable object.

**Did the engineered features help?** Not meaningfully. `AverageCharge` and `LongTermCustomer` are both strongly correlated with existing features already in the model (`TotalCharges`, `tenure`, and `MonthlyCharges`), so Logistic Regression — which already captures linear relationships with those raw columns — gains little extra signal from the derived versions. F1-score is essentially flat (61.33% vs. 61.36%), accuracy and precision dipped very slightly, and recall rose marginally. In short, the new features are **redundant rather than harmful**, and a tree-based model (which can exploit ratio/threshold features like these more directly) or a formal feature-selection step would be needed to see if they add real value.

---

### 💾 Saved Pipeline
The complete trained pipeline (preprocessing + model) was saved as:
```
telco_churn_pipeline.pkl
```
Reload it later with:
```python
import joblib
pipeline = joblib.load("telco_churn_pipeline.pkl")
predictions = pipeline.predict(new_raw_dataframe)
```
Because preprocessing lives inside the pipeline, `new_raw_dataframe` can be raw, unprocessed data with the same columns as the original dataset — no manual scaling or encoding needed before predicting.

---

### 🛠️ Requirements
```
pandas
numpy
seaborn
matplotlib
scikit-learn
joblib
```

Install with:
```bash
pip install pandas numpy seaborn matplotlib scikit-learn joblib
```

---

### ▶️ How to Run
1. Place `Telco-Customer-Churn.csv` at the path used in the script (or update the path).
2. Run the script:
   ```bash
   python task1.py
   ```
3. Review the printed dataset info, feature lists, pipeline training confirmation, evaluation metrics, classification report, and confirmation that `telco_churn_pipeline.pkl` was saved.

---

### 📁 Suggested Project Structure
```
Week 4/Task 1/
│
├── Telco-Customer-Churn.csv
├── task1.py
├── telco_churn_pipeline.pkl
└── README.md
```

---

### ✅ Summary
This task consolidates the Week 3 churn model into a single, production-ready `Pipeline` object combining a `ColumnTransformer` (scaling + one-hot encoding) with Logistic Regression, confirms it reproduces the earlier manual results (73.60% vs. 73.81% accuracy — a negligible difference), adds two engineered features (`AverageCharge`, `LongTermCustomer`) that turn out to be largely redundant with existing columns, and persists the fully fitted pipeline to disk with `joblib` for easy reuse on new data.