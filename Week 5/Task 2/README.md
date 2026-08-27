# Neurofive ML Internship — Week 5, Task 2
## Deploying the Telco Churn Model as a Streamlit App

### 📌 Overview
This task turns the saved Telco Customer Churn pipeline (from Week 4, Task 1) into an interactive **Streamlit web app**. Users fill in a customer's details through form fields, click "Predict Churn," and get an instant prediction — with all preprocessing handled automatically inside the saved pipeline.

---

### 🎯 Objectives
- Reuse the best-performing saved model (`telco_churn_pipeline.pkl`, saved with `joblib`)
- Build a Streamlit app with input fields for the key features and a "Predict" button
- Load the saved model inside the app and display the prediction to the user
- Deploy the app for free (Streamlit Community Cloud or Hugging Face Spaces)

---

### 🤖 Model Used
`telco_churn_pipeline.pkl` — the full end-to-end **Logistic Regression pipeline** built in Week 4, Task 1 (`ColumnTransformer` with `StandardScaler` for numeric features + `OneHotEncoder` for categorical features, chained with `LogisticRegression(class_weight="balanced")`). Because preprocessing lives inside the pipeline, the app only needs to pass in a raw, unprocessed DataFrame — no manual scaling or encoding required at prediction time.

---

### ⚙️ How the App Works

1. **Load the Model**
   `joblib.load("telco_churn_pipeline.pkl")` loads the entire fitted pipeline once when the app starts.

2. **Page Setup**
   `st.set_page_config()` sets the page title and icon; a title and short instruction are shown at the top.

3. **Customer Information Section**
   Input widgets (`st.selectbox`, `st.number_input`) collect every feature the model was trained on:
   - Demographics: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
   - Service usage: `tenure`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`

4. **Account Information Section**
   `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`.

5. **Predict Button**
   When "🔮 Predict Churn" is clicked:
   - **Recreates the two engineered features** the pipeline expects, exactly as they were built during training:
     - `AverageCharge` = `TotalCharges / tenure` (set to `0` if `tenure == 0`, avoiding division by zero)
     - `LongTermCustomer` = `1` if `tenure >= 12` else `0`
   - Assembles all inputs into a single-row `pandas.DataFrame` with the same column names the pipeline was trained on
   - Calls `model.predict(input_data)` — the pipeline internally scales, encodes, and predicts in one step
   - Displays the result: a red warning (`st.error`) if churn is predicted, or a green success message (`st.success`) if not

---

### ✅ Correctness Check

The app's input `DataFrame` was compared column-by-column against the feature lists the training pipeline expects:

| Pipeline expects | App provides | Match |
|---|---|:---:|
| Numeric: `SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`, `AverageCharge`, `LongTermCustomer` | All present, same names | ✅ |
| Categorical: `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod` | All present, same names | ✅ |
| Engineered feature logic (`AverageCharge`, `LongTermCustomer`) | Recomputed identically to training script, including the `tenure == 0` guard | ✅ |

Since `ColumnTransformer` selects columns by name rather than position, column order in `input_data` doesn't matter — only that every expected name is present with a valid value, which it is. The app is correctly wired to the saved pipeline.

---

### 🛠️ Requirements
Create a `requirements.txt` with:
```
streamlit
pandas
scikit-learn
joblib
```

Install locally with:
```bash
pip install streamlit pandas scikit-learn joblib
```

---

### ▶️ Running the App Locally
1. Place `telco_churn_pipeline.pkl` in the same folder as the app script (e.g. `app.py`).
2. Run:
   ```bash
   streamlit run app.py
   ```
3. Streamlit opens automatically in your browser (typically at `http://localhost:8501`). Fill in the customer details and click **🔮 Predict Churn**.

---

### 🚀 Deploying for Free (Streamlit Community Cloud)

1. **Push to GitHub**
   Create a public GitHub repo containing:
   - `app.py` (the Streamlit script)
   - `telco_churn_pipeline.pkl` (the saved model)
   - `requirements.txt`

2. **Sign in to Streamlit Community Cloud**
   Go to [streamlit.io/cloud](https://streamlit.io/cloud) and sign in with your GitHub account.

3. **Create a New App**
   Click "New app," select the repo, branch, and set the main file path to `app.py`.

4. **Deploy**
   Click "Deploy." Streamlit Cloud installs everything from `requirements.txt` and launches the app — you'll get a public URL like `https://your-app-name.streamlit.app` to share.

5. **Redeploying After Changes**
   Any push to the connected GitHub branch automatically triggers a redeploy.

**Alternative — Hugging Face Spaces:**
1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space), choosing **Streamlit** as the Space SDK.
2. Upload `app.py`, `telco_churn_pipeline.pkl`, and `requirements.txt` (or push via Git).
3. The Space builds automatically and hosts the app at `https://huggingface.co/spaces/<username>/<space-name>`.

> ⚠️ Note: `.pkl` files created with a given version of `scikit-learn` should be loaded with a matching (or compatible) version — pin the exact `scikit-learn` version used during training in `requirements.txt` to avoid deployment errors from version mismatches.

---

### 📁 Suggested Project Structure
```
Week 5/Task 2/
│
├── app.py
├── telco_churn_pipeline.pkl
├── requirements.txt
└── README.md
```

---

### ✅ Summary
This task packages the Week 4 churn model into a self-contained, user-friendly Streamlit app: form inputs mirror every feature the pipeline expects, engineered features are recreated with the same logic used during training, and a single button click runs the full preprocessing-plus-prediction pipeline and shows a clear churn/no-churn verdict. The app is ready to deploy as-is to Streamlit Community Cloud or Hugging Face Spaces for free public access.