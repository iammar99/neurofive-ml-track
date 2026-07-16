# ==========================================
#         Importing Libraries
# ==========================================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

 
# ==========================================
#         Importing DataSet
# ==========================================


data = pd.read_csv("G:\\Internships\\Neurofive ML\\Week 1\\Task 1\\Titanic-Dataset.csv")



# ==========================================
#               Tasks
# ==========================================



# Checking how many null values 


data.isna().sum()


# Age            177
# Cabin          687
# Embarked         2

# Embarked filled with mode or we can also drop because it is only 2 values will not effect our data


embarked_mode = data["Embarked"].mode()[0]
data["Embarked"] = data["Embarked"].fillna(embarked_mode)

# Detecting Outliers in Age

sns.boxplot(x=data['Age'])
plt.title('Distribution of Passenger Ages')
plt.show()


data["Age"].min() # 0.42
data["Age"].max() # 80
data["Age"].median() # 28.0
data["Age"].mean() # 30.0

(data["Age"] < 1).sum() # There are 7 values less than 1
data[data["Age"] < 1]["Age"]


# Dropping all these outliers

index_to_drop = data[data["Age"] < 1].index
data["Age"] = data["Age"].drop(index_to_drop)




# Filling NA of Age with mean

mean_age = np.ceil(data["Age"].mean())
data["Age"] = data["Age"].fillna(mean_age)


# Filling NA of Cabin with "Unkown"


data["Cabin"].mode()
data["Cabin"] = data["Cabin"].fillna("Unkown")




# ==========================================
#             visualizations 
# ==========================================




# Ensure seaborn styles are applied
sns.set_theme(style="whitegrid")

# Create a figure with a 2x2 grid layout for all 4 plots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# ==========================================
# 1. Histogram (Top-Left)
# ==========================================
sns.histplot(
    data=data, x="Age", kde=True, bins=30, color="skyblue", ax=axes[0, 0]
)
axes[0, 0].set_title("Distribution of Passenger Ages", fontsize=14)
axes[0, 0].set_xlabel("Age", fontsize=12)
axes[0, 0].set_ylabel("Count", fontsize=12)

# ==========================================
# 2. Boxplot (Top-Right)
# ==========================================
sns.boxplot(data=data, x="Pclass", y="Fare", palette="Set2", ax=axes[0, 1])
axes[0, 1].set_ylim(0, 150)  # Limiting y-axis to see the boxes clearly
axes[0, 1].set_title("Ticket Fare Distribution by Class", fontsize=14)
axes[0, 1].set_xlabel("Passenger Class (Pclass)", fontsize=12)
axes[0, 1].set_ylabel("Fare ($)", fontsize=12)

# ==========================================
# 3. Bar Chart (Bottom-Left)
# ==========================================
sns.countplot(
    data=data, x="Sex", hue="Survived", palette="RdYlBu", ax=axes[1, 0]
)
axes[1, 0].set_title("Survival Count by Gender", fontsize=14)
axes[1, 0].set_xlabel("Gender", fontsize=12)
axes[1, 0].set_ylabel("Passenger Count", fontsize=12)
axes[1, 0].legend(title="Survived", labels=["No", "Yes"])

# ==========================================
# 4. Correlation Heatmap (Bottom-Right)
# ==========================================
# Select only numerical columns for correlation
numerical_cols = data.select_dtypes(include=["float64", "int64"])
corr_matrix = numerical_cols.corr()

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5,
    ax=axes[1, 1],
)
axes[1, 1].set_title("Correlation Heatmap of Numerical Features", fontsize=14)

# Adjust layout so plots don't overlap
plt.tight_layout()

# Display the combined plots
plt.show()



# ==========================================
#             Correlation
# ==========================================



correlation_matrix = data.corr(numeric_only=True)


# Survival is mostly affected by PClass because it is correlating with survival with -0.33 maximum than all others
