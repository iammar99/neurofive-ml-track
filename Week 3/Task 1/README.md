# Neurofive ML Internship — Week 3, Task 1
## Telco Customer Churn Prediction (Decision Tree vs. Logistic Regression)

### 📌 Overview
This task explores the **Telco Customer Churn** dataset to understand what drives customers to leave, then builds and compares two classification models — a **Decision Tree** and **Logistic Regression** — to predict churn. It covers EDA, preprocessing (numeric + categorical + class imbalance handling), model comparison, and feature importance analysis.

---

### 🎯 Objectives
- Use the Telco Customer Churn dataset (from Kaggle)
- Perform EDA to see which features correlate with churn
- Train a Decision Tree Classifier and a Logistic Regression model, and compare them
- Handle categorical variables and class imbalance
- Identify the top 3 features driving churn using `.feature_importances_`

---

### 🗂️ Dataset
`Telco-Customer-Churn.csv` — **7,043 rows × 21 columns**, no missing values and no duplicate rows. Contains customer account, service, and billing information, with the target column `Churn` (`Yes`/`No`). Key features include `tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`, `PaymentMethod`, and various service subscription columns.

**Churn class distribution:**

| Class | Count | Percentage |
|-------|:-----:|:----------:|
| No (stayed)  | 5,174 | 73.46% |
| Yes (churned) | 1,869 | 26.54% |

The dataset is **imbalanced**, with churners making up roughly a quarter of all customers.

---

### ⚙️ Workflow

1. **Import Libraries**
   `pandas`, `numpy`, `matplotlib`, `seaborn`, and modules from `sklearn` for preprocessing, pipelines, modeling, and evaluation.

2. **Load Dataset & Inspect**
   Confirmed shape (7,043 × 21), column names, data types, zero missing values, and zero duplicate rows.

3. **Exploratory Data Analysis (EDA)**
   - Converted `TotalCharges` to numeric (coercing invalid entries to `NaN` — 11 rows became `NaN` out of 7,043, per the `.describe()` count of 7,032 valid values)
   - Checked churn class distribution (above)
   - Boxplots: **Churn vs. Tenure** and **Churn vs. Monthly Charges**
   - Countplots: **Churn by Contract Type** and **Churn by Payment Method**
   - Correlation heatmap of `tenure`, `MonthlyCharges`, `TotalCharges`, and `Churn` (numerically encoded)

4. **Data Preparation**
   - Dropped the non-predictive `customerID` column
   - Re-cast `TotalCharges` to numeric
   - Split features (`X`) from target (`y`, mapped `No → 0`, `Yes → 1`)
   - Auto-identified numeric features: `SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`
   - Auto-identified 15 categorical features: `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`

5. **Train/Test Split**
   80/20 split with `stratify=y` to preserve the churn ratio (`random_state=42`) → **5,634 training samples**, **1,409 testing samples**.

6. **Preprocessing Pipeline**
   - **Numeric features:** median imputation + `StandardScaler`
   - **Categorical features:** most-frequent imputation + `OneHotEncoder(drop="first", handle_unknown="ignore")`
   - Combined via `ColumnTransformer`, wrapped in an `sklearn` `Pipeline` for each model, so preprocessing and modeling happen together with no data leakage.

7. **Class Imbalance Handling**
   Both models were trained with `class_weight="balanced"`, which re-weights the loss function to compensate for churners being the minority class — without manually oversampling/undersampling the data.

8. **Model Training & Evaluation**
   - **Decision Tree Classifier** (`max_depth=5`, `class_weight="balanced"`)
   - **Logistic Regression** (`max_iter=1000`, `class_weight="balanced"`)

9. **Model Comparison**
   Combined both models' metrics into a single comparison table.

10. **Feature Importance**
    Extracted `.feature_importances_` from the trained Decision Tree, matched to the one-hot encoded feature names, and identified the top 3 features driving churn.

---

### 📊 EDA Findings

- **Tenure vs. Churn:** Customers who churn tend to have shorter tenure — newer customers leave more.
- **Monthly Charges vs. Churn:** Customers who churn tend to have higher monthly charges.
- **Contract Type vs. Churn:** Month-to-month customers churn far more than one- or two-year contract customers.
- **Payment Method vs. Churn:** Customers paying by electronic check churn more than those on automatic payment methods.
- **Correlation matrix:**

| | tenure | MonthlyCharges | TotalCharges | Churn |
|---|:---:|:---:|:---:|:---:|
| **tenure** | 1.000 | 0.248 | 0.826 | **-0.352** |
| **MonthlyCharges** | 0.248 | 1.000 | 0.651 | **0.193** |
| **TotalCharges** | 0.826 | 0.651 | 1.000 | **-0.199** |

`tenure` has the strongest relationship with churn (negative — longer-tenured customers churn less), followed by `MonthlyCharges` (positive — pricier plans churn more).

*(See the boxplots, countplots, and correlation heatmap generated by the script.)*

---

### 📊 Results

**Decision Tree — Results**

| Metric | Value |
|--------|:-----:|
| Accuracy | 73.46% |
| Precision | 50.00% |
| Recall | 80.75% |
| F1 Score | 61.76% |

Classification report:

| Class | Precision | Recall | F1-score | Support |
|-------|:---------:|:------:|:--------:|:-------:|
| 0 (No churn) | 0.91 | 0.71 | 0.80 | 1035 |
| 1 (Churn) | 0.50 | 0.81 | 0.62 | 374 |

Confusion matrix:
```
[[733 302]
 [ 72 302]]
```

**Logistic Regression — Results**

| Metric | Value |
|--------|:-----:|
| Accuracy | 73.81% |
| Precision | 50.43% |
| Recall | 78.34% |
| F1 Score | 61.36% |

Classification report:

| Class | Precision | Recall | F1-score | Support |
|-------|:---------:|:------:|:--------:|:-------:|
| 0 (No churn) | 0.90 | 0.72 | 0.80 | 1035 |
| 1 (Churn) | 0.50 | 0.78 | 0.61 | 374 |

**Model Comparison**

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|:--------:|:---------:|:------:|:--------:|
| Decision Tree | 73.46% | 50.00% | 80.75% | 61.76% |
| Logistic Regression | 73.81% | 50.43% | 78.34% | 61.36% |

The two models perform **almost identically overall**. Logistic Regression edges out slightly on accuracy and precision, while the Decision Tree has a slightly higher recall — meaning it catches a bit more of the actual churners, at the cost of a few more false alarms. Both models were tuned toward recall (via `class_weight="balanced"`), which is why precision is noticeably lower than recall for the churn class in both cases — an intentional trade-off, since missing a real churner is usually costlier than flagging a loyal customer.

---

### 🌟 Top 3 Features Driving Churn

| Rank | Feature | Importance |
|------|---------|:----------:|
| 1 | Contract = Two year | 0.405 |
| 2 | Contract = One year | 0.253 |
| 3 | InternetService = Fiber optic | 0.112 |

**Interpretation:** Contract type dominates the Decision Tree's decisions — together, the "Two year" and "One year" contract flags account for roughly **66%** of total feature importance. This lines up with the EDA: locked-in customers on longer contracts are far less likely to churn, so the tree splits heavily on contract type early on. Having fiber-optic internet service is the third strongest driver, likely reflecting that fiber customers tend to pay more and have more service issues, both of which push churn risk up.

*(See the top-3 feature importance bar plot generated by the script.)*

---

### ⚠️ A Note on Class Imbalance

The `Churn` target is imbalanced — 73.46% "No" vs. 26.54% "Yes". A model that just predicted "No churn" for every customer would score ~73% accuracy while catching **zero** actual churners, which is exactly the problem this dataset is meant to solve. This is handled here — but not fully fixed — by using `class_weight="balanced"` in both models, which penalizes misclassifying the minority (churn) class more heavily during training. That's reflected in the results: recall on the churn class (78–81%) is much higher than it would be with default class weights, at the cost of lower precision (~50%). Techniques like SMOTE, random oversampling, or undersampling were **not** applied here and would be a reasonable next step to push performance further.

---

### 🔧 Note on the Script
There's a stray leading indent on the `from sklearn.metrics import (...)` line in the imports block, which will raise an `IndentationError` in Python. Remove the extra leading whitespace so it's flush with the other `import`/`from` statements above it.

---

### 🛠️ Requirements
```
pandas
numpy
seaborn
matplotlib
scikit-learn
```

Install with:
```bash
pip install pandas numpy seaborn matplotlib scikit-learn
```

---

### ▶️ How to Run
1. Download `Telco-Customer-Churn.csv` from Kaggle and place it at the path used in the script (or update the path).
2. Run the script:
   ```bash
   python task1.py
   ```
3. Review the printed EDA summaries, model metrics, classification reports, confusion matrix, model comparison table, and top-3 feature importance plot.

---

### 📁 Suggested Project Structure
```
Week 3/Task 1/
│
├── Telco-Customer-Churn.csv
├── task1.py
└── README.md
```

---

### ✅ Summary
Both models land around **73–74% accuracy** with near-identical F1-scores (~0.61–0.62). Contract type — especially having a one- or two-year contract — is by far the strongest signal against churn, followed by having fiber-optic internet. Class imbalance was addressed with balanced class weighting, trading some precision for stronger recall on the churn class, since catching at-risk customers matters more than avoiding false alarms in this context.