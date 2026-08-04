# ==========================================
#         Importing Libraries
# ==========================================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix

 
# ==========================================
#         Importing DataSet
# ==========================================


data = pd.read_csv("G:\\Internships\\Neurofive ML\\Week 2\\Task 1\\Titanic_Cleaned.csv")




# ==========================================
#     Encoding categorical columns 
# ==========================================


# Encode categorical columns (e.g., "Sex", "Embarked") using OneHotEncoder or pd.get_dummies()

df_encoded = pd.get_dummies(data, columns=['Sex', 'Embarked','Ticket','Cabin'], drop_first=True)


# ==========================================
#           Splitting dataset
# ==========================================


# Use scikit-learn to split your cleaned dataset into training and test sets using train_test_split

X = df_encoded.drop(columns=['Survived','Name']) 
y = df_encoded['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training features shape: {X_train.shape}")
print(f"Testing features shape: {X_test.shape}")





# ==========================================
#           Selecting a Model
# ==========================================


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)





# ==========================================
#    Testing Accuracy & Confusion Matrix
# ==========================================



y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")



# Model Accuracy: 81.01%


cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)


# ==========================================
#           Visualization
# ==========================================

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Did Not Survive', 'Survived'], 
            yticklabels=['Did Not Survive', 'Survived'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()



