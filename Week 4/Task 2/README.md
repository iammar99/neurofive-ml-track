# Neurofive ML Internship — Week 4, Task 2
## Ensemble Models: Random Forest vs. XGBoost (Telco Customer Churn)

### 📌 Overview
This task extends the Telco Customer Churn pipeline with two **ensemble models** — `RandomForestClassifier` and `XGBClassifier` — compares them against the earlier single Logistic Regression model, and analyzes/plots feature importances from both ensembles to see which features each considers most predictive.

---

### 🎯 Objectives
- Install and use `xgboost`
- Train a `RandomForestClassifier` and an `XGBClassifier` on the churn dataset
- Compare their performance against the earlier single model (Logistic Regression)
- Plot feature importances for both ensemble models and compare which features each ranks highest
- Explain, in a few sentences, how Random Forest and XGBoost differ in how they combine models
- Update the comparison table (model, metric, score) for the GitHub repo

---

### 🗂️ Dataset
`Telco-Customer-Churn.csv` — **7,043 rows × 21 columns**. Same cleaning and feature engineering as Week 4 Task 1: `TotalCharges` converted to numeric, `customerID` dropped, plus two engineered features:
- **`AverageCharge`** = `TotalCharges / tenure`
- **`LongTermCustomer`** = `1` if `tenure >= 12` else `0`

Target: `Churn` mapped to `0` (No) / `1` (Yes).

---

### ⚙️ Workflow

1. **Import Libraries**
   `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn` (pipeline, compose, preprocessing, impute, ensemble, linear_model, metrics), and `xgboost` (`XGBClassifier`).

2. **Load & Clean Data**
   Loaded the dataset, converted `TotalCharges` to numeric, dropped `customerID`.

3. **Feature Engineering**
   Added `AverageCharge` and `LongTermCustomer` (same as Task 1).

4. **Feature Types**
   - **Numeric (6):** `SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`, `AverageCharge`, `LongTermCustomer`
   - **Categorical (15):** `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`

5. **Train/Test Split**
   80/20 split, `stratify=y`, `random_state=42`.

6. **Shared Preprocessing**
   A single `ColumnTransformer` used by all three models:
   - **Numeric:** median imputation (no scaling this time, since tree-based models don't need feature scaling)
   - **Categorical:** most-frequent imputation + `OneHotEncoder(drop="first", handle_unknown="ignore")`

7. **Three Models, Same Preprocessing**
   - **Logistic Regression** (`max_iter=1000`, `class_weight="balanced"`) — the baseline single model
   - **Random Forest** (`n_estimators=100`, `class_weight="balanced"`, `random_state=42`)
   - **XGBoost** (`n_estimators=100`, `max_depth=3`, `learning_rate=0.1`, `eval_metric="logloss"`, `random_state=42`)

   Each wrapped in its own `Pipeline` with the shared `preprocessor`, trained, and evaluated with a common `evaluate_model()` helper (accuracy, precision, recall, F1-score, classification report).

8. **Feature Importance**
   Extracted `.feature_importances_` from both the Random Forest and XGBoost models (mapped back to one-hot encoded feature names) and plotted the top 10 for each.

9. **Compare Top Features**
   Printed and compared the top-5 most important features from Random Forest vs. XGBoost side by side.

---

### 📊 Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|:--------:|:---------:|:------:|:--------:|
| Logistic Regression | 73.53% | 50.08% | 79.14% | 61.35% |
| Random Forest | 79.21% | 64.11% | 49.20% | 55.67% |
| XGBoost | **80.20%** | **66.32%** | 51.60% | 58.05% |

**Takeaways:**
- **XGBoost is the strongest overall model**, with the highest accuracy (80.20%), precision (66.32%), and F1-score (58.05%) among the three.
- **Random Forest** performs similarly to XGBoost — noticeably more accurate and precise than Logistic Regression, but with lower recall.
- **Logistic Regression** has by far the best **recall** (79.14%) — it catches the most actual churners — but at the cost of much lower precision (50.08%), meaning it also flags a lot of customers who don't actually churn. This is a direct result of `class_weight="balanced"`, which the tree ensembles weren't given here (only the Random Forest used it; XGBoost did not).
- **Trade-off:** if the business priority is "catch as many potential churners as possible, even with false alarms," Logistic Regression wins on recall. If the priority is "when we flag a customer as at-risk, be right more often" (e.g., to target a limited retention budget efficiently), XGBoost or Random Forest are the better picks.

---

### 🌟 Feature Importance — Random Forest (Top 10)

| Rank | Feature | Importance |
|------|---------|:----------:|
| 1 | TotalCharges | 0.142 |
| 2 | tenure | 0.138 |
| 3 | AverageCharge | 0.121 |
| 4 | MonthlyCharges | 0.118 |
| 5 | Contract = Two year | 0.048 |
| 6 | InternetService = Fiber optic | 0.043 |
| 7 | PaymentMethod = Electronic check | 0.037 |
| 8 | LongTermCustomer | 0.037 |
| 9 | OnlineSecurity = Yes | 0.023 |
| 10 | Contract = One year | 0.023 |

### 🌟 Feature Importance — XGBoost (Top 10)

| Rank | Feature | Importance |
|------|---------|:----------:|
| 1 | Contract = Two year | 0.211 |
| 2 | InternetService = Fiber optic | 0.193 |
| 3 | Contract = One year | 0.178 |
| 4 | InternetService = No | 0.078 |
| 5 | PaymentMethod = Electronic check | 0.065 |
| 6 | tenure | 0.054 |
| 7 | StreamingMovies = Yes | 0.036 |
| 8 | PaperlessBilling = Yes | 0.027 |
| 9 | OnlineSecurity = Yes | 0.024 |
| 10 | StreamingTV = Yes | 0.021 |

**Top 5 side by side:**

| Rank | Random Forest | XGBoost |
|------|----------------|---------|
| 1 | TotalCharges | Contract = Two year |
| 2 | tenure | InternetService = Fiber optic |
| 3 | AverageCharge | Contract = One year |
| 4 | MonthlyCharges | InternetService = No |
| 5 | Contract = Two year | PaymentMethod = Electronic check |

**How the two models disagree:** Random Forest leans heavily on the **continuous billing/tenure numbers** (`TotalCharges`, `tenure`, `AverageCharge`, `MonthlyCharges` make up its entire top 4), with categorical features like `Contract` only showing up further down. XGBoost does the opposite — its top 4 are all **categorical service/contract flags**, and the only numeric feature in its top 10 is `tenure` at rank 6. Both models agree that `Contract` type and `tenure` matter, but Random Forest is effectively summarizing churn risk through the raw dollar amounts (which correlate strongly with tenure and contract length anyway — see the Week 3 correlation matrix), while XGBoost's boosting process finds sharper, more direct splits on categorical flags like contract type and internet service, giving it a more "human-readable" ranking of *why* someone churns.

*(See the two feature-importance bar plots generated by the script.)*

---

### 🌲 Random Forest vs. XGBoost: How They Combine Models

Random Forest builds many decision trees **independently and in parallel**, each trained on a random subset of the data and features (bagging), and then averages their predictions (or takes a majority vote) to reduce variance and avoid overfitting any single tree. XGBoost instead builds trees **sequentially**, where each new tree is trained specifically to correct the errors (residuals) made by the trees built before it — a technique called gradient boosting. Because Random Forest's trees are independent, it's naturally robust and less prone to overfitting straight out of the box, while XGBoost's error-correcting sequence tends to squeeze out higher accuracy but is more sensitive to hyperparameters like learning rate and tree depth. In this task, that difference shows up directly: XGBoost's boosting process edged out Random Forest's bagging on every metric (80.20% vs. 79.21% accuracy), though both still trail Logistic Regression on recall.

---

### 🔧 Note on the Script
Training the Logistic Regression pipeline raised a `ConvergenceWarning` ("lbfgs failed to converge after 1000 iteration(s)"). This happened because, unlike Week 4 Task 1, the numeric preprocessing pipeline here only imputes — it dropped `StandardScaler`. Since Logistic Regression is scale-sensitive, the unscaled `TotalCharges`/`MonthlyCharges` values (which range into the thousands) slow convergence. This doesn't affect Random Forest or XGBoost (tree-based models don't need scaled features), but if you want to silence the warning and get a cleaner Logistic Regression fit, add `StandardScaler()` back into the numeric pipeline or raise `max_iter` further.

---

### 🛠️ Requirements
```
pandas
numpy
seaborn
matplotlib
scikit-learn
xgboost
```

Install with:
```bash
pip install pandas numpy seaborn matplotlib scikit-learn xgboost
```

---

### ▶️ How to Run
1. Place `Telco-Customer-Churn.csv` at the path used in the script (or update the path).
2. Run the script:
   ```bash
   python task2.py
   ```
3. Review the printed model comparison table, per-model classification reports, top-10 feature importance tables, and the two feature-importance bar plots.

---

### 📁 Suggested Project Structure
```
Week 4/Task 2/
│
├── Telco-Customer-Churn.csv
├── task2.py
└── README.md
```

---

### 📋 Comparison Table (for GitHub repo)

| Model | Metric | Score |
|-------|--------|:-----:|
| Logistic Regression | Accuracy | 73.53% |
| Logistic Regression | Precision | 50.08% |
| Logistic Regression | Recall | 79.14% |
| Logistic Regression | F1 Score | 61.35% |
| Random Forest | Accuracy | 79.21% |
| Random Forest | Precision | 64.11% |
| Random Forest | Recall | 49.20% |
| Random Forest | F1 Score | 55.67% |
| XGBoost | Accuracy | 80.20% |
| XGBoost | Precision | 66.32% |
| XGBoost | Recall | 51.60% |
| XGBoost | F1 Score | 58.05% |

---

### ✅ Summary
XGBoost edges out Random Forest as the top-performing model overall (80.20% accuracy, 66.32% precision), with both ensembles beating Logistic Regression on accuracy and precision but losing to it on recall. Random Forest's feature importances lean on raw billing/tenure numbers, while XGBoost's lean on contract and service-type categories — different views of the same underlying story: long-term, contracted customers with lower-risk service plans churn the least.