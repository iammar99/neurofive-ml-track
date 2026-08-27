# ==========================================
#     House Price Prediction Project
# ==========================================


# ==========================================
#          Importing Libraries
# ==========================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import joblib


# ==========================================
#          Load Dataset
# ==========================================

print("Loading Dataset...")

df = pd.read_csv(
    "housing.csv"
)

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())


# ==========================================
#          Basic Information
# ==========================================

print("\nDataset Information:")

print(df.info())


print("\nMissing Values:")

print(df.isnull().sum())


print("\nDuplicate Rows:")

print(df.duplicated().sum())


# ==========================================
#              Basic EDA
# ==========================================

print("\nDataset Statistics:")

print(df.describe())


# ==========================================
#        Price Distribution
# ==========================================

plt.figure(figsize=(8, 5))

sns.histplot(
    df["price"],
    bins=30
)

plt.title(
    "Distribution of House Prices"
)

plt.xlabel(
    "House Price"
)

plt.ylabel(
    "Number of Houses"
)

plt.show()


# ==========================================
#          Area vs Price
# ==========================================

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="area",
    y="price"
)

plt.title(
    "Area vs House Price"
)

plt.xlabel(
    "Area"
)

plt.ylabel(
    "House Price"
)

plt.show()


# ==========================================
#        Bedrooms vs Price
# ==========================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="bedrooms",
    y="price"
)

plt.title(
    "Bedrooms vs House Price"
)

plt.xlabel(
    "Bedrooms"
)

plt.ylabel(
    "House Price"
)

plt.show()


# ==========================================
#       Furnishing vs Price
# ==========================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="furnishingstatus",
    y="price"
)

plt.title(
    "Furnishing Status vs House Price"
)

plt.xticks(rotation=15)

plt.show()


# ==========================================
#            Correlation
# ==========================================

numeric_df = df.select_dtypes(
    include=["int64", "float64"]
)

correlation = numeric_df.corr()

print("\nCorrelation with House Price:")

print(
    correlation["price"].sort_values(
        ascending=False
    )
)


plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title(
    "Correlation Matrix"
)

plt.show()


# ==========================================
#        Feature Engineering
# ==========================================

# Feature 1:
# Price-related size information

df["area_per_bedroom"] = (
    df["area"] /
    df["bedrooms"]
)


# Feature 2:
# Total number of rooms

df["total_rooms"] = (
    df["bedrooms"] +
    df["bathrooms"]
)


print("\nNew Features Created:")

print("1. area_per_bedroom")

print("2. total_rooms")


# ==========================================
#          Create X and y
# ==========================================

X = df.drop(
    "price",
    axis=1
)

y = df["price"]


# ==========================================
#   Identify Numerical/Categorical Columns
# ==========================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()


categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()


print("\nNumerical Features:")

print(numeric_features)


print("\nCategorical Features:")

print(categorical_features)


# ==========================================
#          Train/Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("\nTraining Samples:")

print(len(X_train))


print("\nTesting Samples:")

print(len(X_test))


# ==========================================
#          Preprocessing
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[

        (
            "numeric",
            Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        )
                    ),

                    (
                        "scaler",
                        StandardScaler()
                    )
                ]
            ),
            numeric_features
        ),

        (
            "categorical",
            Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        )
                    ),

                    (
                        "encoder",
                        OneHotEncoder(
                            drop="first",
                            handle_unknown="ignore"
                        )
                    )
                ]
            ),
            categorical_features
        )

    ]
)


# ==========================================
#        Linear Regression Model
# ==========================================

linear_model = Pipeline(
    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            LinearRegression()
        )

    ]
)


# ==========================================
#       Random Forest Model
# ==========================================

random_forest = Pipeline(
    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )
        )

    ]
)


# ==========================================
#          Train Models
# ==========================================

print("\nTraining Linear Regression...")

linear_model.fit(
    X_train,
    y_train
)


print("Training Random Forest...")

random_forest.fit(
    X_train,
    y_train
)


# ==========================================
#            Predictions
# ==========================================

linear_predictions = (
    linear_model.predict(
        X_test
    )
)


rf_predictions = (
    random_forest.predict(
        X_test
    )
)


# ==========================================
#       Linear Regression Evaluation
# ==========================================

linear_mae = mean_absolute_error(
    y_test,
    linear_predictions
)

linear_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        linear_predictions
    )
)

linear_r2 = r2_score(
    y_test,
    linear_predictions
)


# ==========================================
#       Random Forest Evaluation
# ==========================================

rf_mae = mean_absolute_error(
    y_test,
    rf_predictions
)

rf_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        rf_predictions
    )
)

rf_r2 = r2_score(
    y_test,
    rf_predictions
)


# ==========================================
#          Model Comparison
# ==========================================

results = pd.DataFrame({

    "Model": [
        "Linear Regression",
        "Random Forest"
    ],

    "MAE": [
        linear_mae,
        rf_mae
    ],

    "RMSE": [
        linear_rmse,
        rf_rmse
    ],

    "R2 Score": [
        linear_r2,
        rf_r2
    ]

})


print("\n==========================================")

print("          MODEL COMPARISON")

print("==========================================")

print(results)


# ==========================================
#          Select Best Model
# ==========================================

if rf_r2 > linear_r2:

    best_model = random_forest

    print(
        "\nBest Model: Random Forest"
    )

else:

    best_model = linear_model

    print(
        "\nBest Model: Linear Regression"
    )


# ==========================================
#            Save Model
# ==========================================

joblib.dump(
    best_model,
    "house_price_model.pkl"
)


print(
    "\nModel saved successfully as:"
)

print(
    "house_price_model.pkl"
)


# ==========================================
#          Model Location
# ==========================================

import os

print(
    "\nModel saved at:"
)

print(
    os.path.abspath(
        "house_price_model.pkl"
    )
)


# ==========================================
#              Final Result
# ==========================================

print("\n==========================================")

print("           FINAL RESULT")

print("==========================================")


if rf_r2 > linear_r2:

    print(
        "Best Model: Random Forest"
    )

    print(
        "Best R2 Score:",
        rf_r2
    )

else:

    print(
        "Best Model: Linear Regression"
    )

    print(
        "Best R2 Score:",
        linear_r2
    )


print(
    "\nProject completed successfully!"
)
