import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

plt.style.use("seaborn-v0_8-whitegrid")

# ============================================================
# 1. DATASET LOADING AND INFORMATION
# ============================================================
data = pd.read_csv(r"/Users/shubhamkumar/Desktop/pp project/Electric_Vehicle_Population_Size_History_By_County_.csv")
print(data.head())
print(data.shape)
print(data.info())
print(data.describe())

# Column names short
data = data.rename(columns={
    "Battery Electric Vehicles (BEVs)": "BEV",
    "Plug-In Hybrid Electric Vehicles (PHEVs)": "PHEV",
    "Electric Vehicle (EV) Total": "EV_Total",
    "Non-Electric Vehicle Total": "Non_EV_Total",
    "Total Vehicles": "Total_Vehicles",
    "Percent Electric Vehicles": "Percent_EV",
    "Vehicle Primary Use": "Vehicle_Use"
})

# ============================================================
# 2. DATA PREPROCESSING AND CLEANING
# ============================================================
# Numbers text me ho sakte hain (jaise "3,620"), unhe number me convert karna
num_cols = ["BEV", "PHEV", "EV_Total", "Non_EV_Total", "Total_Vehicles", "Percent_EV"]
data[num_cols] = data[num_cols].replace("[,%]", "", regex=True).apply(pd.to_numeric, errors="coerce")

# Missing values
print("\nMissing Values:")
print(data.isna().sum())
data = data.dropna()

# Duplicates
print("\nDuplicate rows:", data.duplicated().sum())
data = data.drop_duplicates()

# Inconsistent data (extra space aur different spelling)
data["County"] = data["County"].str.strip().str.title()
data["Vehicle_Use"] = data["Vehicle_Use"].str.strip().str.title()

# Date ko datetime me convert karna
data["Date"] = pd.to_datetime(data["Date"])
print("\nRows after cleaning:", data.shape[0])

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
data["Year"] = data["Date"].dt.year

# ============================================================
# 4. EDA AND PROJECT OBJECTIVES
# ============================================================
latest_data = data[data["Date"] == data["Date"].max()]
monthly = data.groupby("Date")[["BEV", "PHEV", "EV_Total"]].sum()

# Objective 1: Time ke saath total EV kaise badhe?
first_ev = monthly["EV_Total"].iloc[0]
last_ev = monthly["EV_Total"].iloc[-1]
print("\nObjective 1: EV growth =", round((last_ev - first_ev) / first_ev * 100, 1), "%")

plt.figure(figsize=(11, 5))
plt.plot(monthly.index, monthly["EV_Total"], color="#2e8b57", linewidth=2.5)
plt.fill_between(monthly.index, monthly["EV_Total"], color="#2e8b57", alpha=0.2)
plt.title("Objective 1: Total EV Growth Over Time", fontsize=14, fontweight="bold")
plt.xlabel("Date")
plt.ylabel("Total EVs")
plt.ticklabel_format(style="plain", axis="y")
plt.show()

# Objective 2: Kaunse top 10 counties me sabse zyada EV hain?
top_counties = latest_data.groupby("County")["EV_Total"].sum().sort_values(ascending=False).head(10)
print("\nObjective 2: Top Counties")
print(top_counties)

plt.figure(figsize=(10, 6))
bars = plt.barh(top_counties.sort_values().index, top_counties.sort_values().values, color="#4c9be8")
plt.bar_label(bars, fmt="{:,.0f}", padding=4)
plt.title("Objective 2: Top 10 Counties by EV Total", fontsize=14, fontweight="bold")
plt.xlabel("EV Total")
plt.ylabel("County")
plt.show()

# Objective 3: BEV vs PHEV (top 5 counties)
top5 = latest_data.groupby("County")[["BEV", "PHEV"]].sum().loc[top_counties.index[:5]]
print("\nObjective 3: BEV vs PHEV")
print(top5)

top5.plot(kind="bar", figsize=(10, 5), color=["#1f77b4", "#ff7f0e"], edgecolor="white")
plt.title("Objective 3: BEV vs PHEV in Top 5 Counties", fontsize=14, fontweight="bold")
plt.xlabel("County")
plt.ylabel("Number of Vehicles")
plt.xticks(rotation=0)
plt.show()

# Objective 4: Passenger vs Truck me EV distribution
use_data = latest_data.groupby("Vehicle_Use")["EV_Total"].sum()
print("\nObjective 4: EV by Vehicle Use")
print(use_data)

use_time = data.groupby(["Date", "Vehicle_Use"])["EV_Total"].sum().unstack()
use_time.plot(kind="line", figsize=(11, 5), linewidth=2.5, color=["#e74c3c", "#3498db"])
plt.title("Objective 4: Passenger vs Truck EVs Over Time", fontsize=14, fontweight="bold")
plt.xlabel("Date")
plt.ylabel("Total EVs")
plt.ticklabel_format(style="plain", axis="y")
plt.legend(title="Vehicle Use")
plt.show()

# Objective 5: Percent EV har saal kaise badla?
year_percent = data.groupby("Year")["Percent_EV"].mean()
print("\nObjective 5: Average Percent EV by Year")
print(year_percent.round(2))

plt.figure(figsize=(10, 5))
bars = plt.bar(year_percent.index.astype(str), year_percent.values, color="#f4a261")
plt.bar_label(bars, fmt="{:.2f}%", padding=3)
plt.title("Objective 5: Average Percent EV by Year", fontsize=14, fontweight="bold")
plt.xlabel("Year")
plt.ylabel("Percent EV")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(9, 7))
sns.heatmap(data[num_cols + ["Year"]].corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap", fontsize=14, fontweight="bold")
plt.show()

# ============================================================
# 5. FEATURES AND TARGET
# ============================================================
# Target: Vehicle_Use (Passenger ya Truck)
a = LabelEncoder()
data["Vehicle_Use"] = a.fit_transform(data["Vehicle_Use"])
print("\nTarget classes:", list(a.classes_))

X = data[["BEV", "PHEV", "EV_Total", "Non_EV_Total", "Total_Vehicles", "Percent_EV", "Year"]]
y = data["Vehicle_Use"]

# ============================================================
# 6. TRAIN TEST SPLIT AND FEATURE SCALING
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# ============================================================
# 7. MODEL 1 : DECISION TREE
# ============================================================
classifier = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)
classifier.fit(X_train, y_train)
y_pred_dt = classifier.predict(X_test)

print("\n----- DECISION TREE -----")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))
print("\nClassification Report:\n", classification_report(y_test, y_pred_dt))

plt.figure(figsize=(16, 10))
plot_tree(classifier, feature_names=list(X.columns), class_names=list(a.classes_), rounded=True, filled=True)
plt.title("Decision Tree Diagram")
plt.show()

# ============================================================
# 8. MODEL 2 : KNN
# ============================================================
knn = KNeighborsClassifier(
    n_neighbors=5,
    metric='euclidean'  # Euclidean distance
)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

print("\n----- KNN -----")
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))
print("\nClassification Report:\n", classification_report(y_test, y_pred_knn))

# ============================================================
# 9. MODEL 3 : NAIVE BAYES
# ============================================================
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)

print("\n----- NAIVE BAYES -----")
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_nb))
print("\nClassification Report:\n", classification_report(y_test, y_pred_nb))

# ============================================================
# 10. MODEL 4 : LOGISTIC REGRESSION
# ============================================================
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

print("\n----- LOGISTIC REGRESSION -----")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
print("\nClassification Report:\n", classification_report(y_test, y_pred_lr))

# ============================================================
# 11. MODEL COMPARISON
# ============================================================
results = pd.DataFrame({
    "Model": ["Decision Tree", "KNN", "Naive Bayes", "Logistic Regression"],
    "Accuracy": [
        accuracy_score(y_test, y_pred_dt),
        accuracy_score(y_test, y_pred_knn),
        accuracy_score(y_test, y_pred_nb),
        accuracy_score(y_test, y_pred_lr)
    ]
})
print("\nModel Comparison:")
print(results)

plt.figure(figsize=(8, 5))
bars = plt.bar(results["Model"], results["Accuracy"], color=["#2ecc71", "#3498db", "#f39c12", "#e74c3c"])
plt.bar_label(bars, fmt="{:.3f}", padding=3)
plt.title("Model Comparison (Accuracy)", fontsize=14, fontweight="bold")
plt.ylabel("Accuracy")
plt.ylim(0, 1.1)
plt.show()

# ============================================================
# 12. FINAL CONCLUSION
# ============================================================
best_model = results.loc[results["Accuracy"].idxmax(), "Model"]
print("\n========== FINAL CONCLUSION ==========")
print("Best model:", best_model, "with accuracy", round(results["Accuracy"].max(), 4))
print("Top county for EV:", top_counties.index[0])
print("\nRecommendations:")
print("1. Build more charging stations in top counties.")
print("2. Give incentives in counties with low EV adoption.")
print("3. Encourage EV trucks for commercial use.")