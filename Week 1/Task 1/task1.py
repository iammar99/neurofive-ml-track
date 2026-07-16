# ==========================================
#         Importing Libraries
# ==========================================
import pandas as pd


 
# ==========================================
#         Importing DataSet
# ==========================================


data = pd.read_csv("G:\\Internships\\Neurofive ML\\Week 1\\Task1\\Titanic-Dataset.csv")



# ==========================================
#               Tasks
# ==========================================

# how many rows/columns, which columns have missing values, and which are categorical vs numerical

data.head()
data.info()
data.describe()


data.isna().sum()


# [891 rows x 12 columns]

# PassengerId      0
# Survived         0
# Pclass           0
# Name             0
# Sex              0
# Age            177
# SibSp            0
# Parch            0
# Ticket           0
# Fare             0
# Cabin          687
# Embarked         2





# categorical :- 

# Survived,pclass,embarked,Sex


# Numerical :- 

# age,fare,sibsp,parch



