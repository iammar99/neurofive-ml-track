# ==========================================
#         Importing Libraries
# ==========================================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

 
# ==========================================

#         Importing DataSet
# ==========================================

data = pd.read_csv('G:\\Internships\\Neurofive ML\\Week 2\\Task 2\\Housing.csv')



# ==========================================
#               Tasks
# ==========================================


# ------------------| Encoding categorical data |------------------

data = pd.get_dummies(data,columns=["mainroad","guestroom","basement","hotwaterheating","airconditioning","prefarea"],drop_first=True)
data = pd.get_dummies(data,columns=["furnishingstatus"],dtype=int)
data.columns


# ------------------| Selecting most affective features |------------------


# Select 3-5 features you believe most affect price (e.g., square footage, number of rooms, location)

correlation = data.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.show()


y = data["price"]
x = data[["area","airconditioning_yes","bathrooms","stories","parking","bedrooms","prefarea_yes","mainroad_yes"]]



# ------------------| Training the model |------------------



# Train a Linear Regression model using scikit-learn


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)


# ------------------| Evaluting Performance |------------------

# Evaluate performance using RMSE (Root Mean Squared Error) and R² score

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")





# ------------------| Visulaization |------------------



# Plot predicted vs. actual prices on a scatter plot to visually check your model's quality


plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='blue', edgecolors='k', label='Predictions')

min_val = min(min(y_test), min(y_pred))
max_val = max(max(y_test), max(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Ideal Fit')

plt.title('Actual vs. Predicted Prices', fontsize=14, fontweight='bold')
plt.xlabel('Actual Prices', fontsize=12)
plt.ylabel('Predicted Prices', fontsize=12)
plt.legend()
plt.tight_layout()
plt.show()
