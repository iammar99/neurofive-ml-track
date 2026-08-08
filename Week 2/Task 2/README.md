# Neurofive ML Internship — Week 2, Task 2
## House Price Prediction using Linear Regression

### 📌 Overview
This task builds a **Linear Regression** model to predict house prices using the Housing dataset. It covers feature selection based on correlation analysis, encoding categorical variables, training a regression model, and evaluating it using RMSE and R² score.

---

### 🎯 Objectives
- Use a housing dataset with price and property features
- Select 3–5 features believed to most affect price
- Train a Linear Regression model using scikit-learn
- Evaluate performance using RMSE (Root Mean Squared Error) and R² score
- Plot predicted vs. actual prices to visually assess model quality
- Explain the R² score in plain English

---

### 🗂️ Dataset
`Housing.csv` — a housing dataset containing the target column `price` along with features such as `area`, `bedrooms`, `bathrooms`, `stories`, `parking`, `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea`, and `furnishingstatus`.

---

### ⚙️ Workflow

1. **Import Libraries**
   `pandas`, `numpy`, `seaborn`, `matplotlib`, and relevant modules from `sklearn`.

2. **Load Dataset**
   Read the housing CSV file into a DataFrame.

3. **Encode Categorical Columns**
   - Binary yes/no columns (`mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea`) encoded with `pd.get_dummies(drop_first=True)`.
   - `furnishingstatus` (multi-category) one-hot encoded with `pd.get_dummies()`.

4. **Feature Selection via Correlation**
   Generated a correlation heatmap to identify which features most strongly relate to `price`, then selected:
   - `area`
   - `bathrooms`
   - `stories`
   - `parking`
   - `bedrooms`
   - `airconditioning_yes`
   - `prefarea_yes`
   - `mainroad_yes`

5. **Train/Test Split**
   Split into 80% training and 20% testing data using `train_test_split(random_state=42)`.

6. **Model Training**
   Trained a `LinearRegression()` model on the training set.

7. **Evaluation**
   - Predicted prices on the test set
   - Calculated **RMSE** with `mean_squared_error` (square-rooted) and **R²** with `r2_score`
   - Plotted **Actual vs. Predicted Prices** with a red dashed "ideal fit" line for visual comparison

---

### 📊 Feature Correlation Heatmap

The heatmap shows how strongly each numeric/encoded feature correlates with `price` and with each other. `area` (0.54), `bathrooms` (0.52), `airconditioning_yes` (0.45), `stories` (0.42), `parking` (0.38), and `mainroad_yes` (0.30) showed the strongest positive correlation with price, guiding the feature selection above.

*(See `correlation.png`)*

---

### 📈 Actual vs. Predicted Prices

The scatter plot compares actual test-set prices against the model's predicted prices. Points closer to the red "Ideal Fit" line indicate more accurate predictions. The model tracks the general upward trend well for low-to-mid range prices, but tends to **underpredict** for higher-priced houses (points falling below the line at the top-right), suggesting the linear model doesn't fully capture the dynamics driving very expensive homes.

*(See `scatterplot.png`)*

---

### 📊 Results

| Metric | Value |
|--------|-------|
| RMSE   | *(insert the value printed by your script, e.g. `1,150,000`)* |
| R² Score | *(insert the value printed by your script, e.g. `0.65`)* |

> ⚠️ Replace the placeholders above with the exact numbers printed by `print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")` and `print(f"R² Score: {r2:.4f}")` when you run the script.

---

### 🗣️ What Does the R² Score Mean? (Plain English)

The R² score tells us **how well our model's predictions match the real house prices**, on a scale from 0 to 1. An R² of, say, 0.65 means our model can explain about **65% of the reasons why house prices go up or down**, based on the features we gave it (like area, bathrooms, and air conditioning). The remaining 35% is influenced by things our model doesn't account for — like exact location, renovation quality, or market conditions. In short, the closer R² is to 1, the more trustworthy the model's price predictions are; a score like this means the model is **reasonably useful but not perfect**, so predictions should be treated as good estimates rather than exact figures.


---

### 📁 Suggested Project Structure
```
Week 2/Task 2/
│
├── Housing.csv
├── task2.py
├── correlation.png
├── scatterplot.png
└── README.md
```

---

### ✅ Summary
This task demonstrates a full regression pipeline — encoding categorical features, selecting predictors through correlation analysis, training a Linear Regression model, and evaluating it with RMSE and R² — to estimate house prices and visually assess prediction quality against actual values.