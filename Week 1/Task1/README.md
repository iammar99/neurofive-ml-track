# Task 1 — Titanic Dataset: Exploratory Data Analysis

## Overview
This task covers the basics of loading, inspecting, and summarizing a real-world dataset using Python, pandas, and NumPy. The dataset used is the Titanic dataset from Kaggle's "Titanic - Machine Learning from Disaster" competition.

## Tools Used
- Python
- Rstudio
- pandas
- NumPy

## Steps Performed
1. Installed Python, RStudio, pandas, and NumPy.
2. Downloaded the Titanic dataset from Kaggle.
3. Loaded the dataset using `pandas.read_csv()`.
4. Inspected the data using `.info()`, `.describe()`, and `.head()`.
5. Identified dataset shape, missing values, and column types.
6. Summarized findings in a markdown cell within the notebook.

## Dataset Summary

**Shape:** 891 rows × 12 columns

**Missing Values:**

| Column      | Missing Values |
|-------------|----------------|
| PassengerId | 0 |
| Survived    | 0 |
| Pclass      | 0 |
| Name        | 0 |
| Sex         | 0 |
| Age         | 177 |
| SibSp       | 0 |
| Parch       | 0 |
| Ticket      | 0 |
| Fare        | 0 |
| Cabin       | 687 |
| Embarked    | 2 |

**Categorical columns:** Survived, Pclass, Embarked, Sex

**Numerical columns:** Age, Fare, SibSp, Parch

## Data Story (Findings)
The Titanic dataset contains 891 passenger records across 12 columns. Two columns — Age and Cabin — have significant missing data, with Cabin missing in over 75% of rows, making it unreliable for direct analysis. Age has moderate missingness (~20%) and will likely need imputation. Embarked has only 2 missing values and can be handled easily. The dataset has a mix of categorical (Survived, Pclass, Sex, Embarked) and numerical (Age, Fare, SibSp, Parch) features, setting up a solid foundation for further cleaning and feature engineering.

