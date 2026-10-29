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
# LOAD & PREPROCESS DATA
# ============================================================
data = pd.read_csv("Electric_Vehicle_Population_Size_History_By_County_.csv")
data = data.rename(columns={
    "Battery Electric Vehicles (BEVs)": "BEV",
    "Plug-In Hybrid Electric Vehicles (PHEVs)": "PHEV",
    "Electric Vehicle (EV) Total": "EV_Total",
    "Non-Electric Vehicle Total": "Non_EV_Total",
    "Total Vehicles": "Total_Vehicles",
    "Percent Electric Vehicles": "Percent_EV",
    "Vehicle Primary Use": "Vehicle_Use"
})
num_cols = ["BEV", "PHEV", "EV_Total", "Non_EV_Total", "Total_Vehicles", "Percent_EV"]
data[num_cols] = data[num_cols].replace("[,%]", "", regex=True).apply(pd.to_numeric, errors="coerce")
data = data.dropna().drop_duplicates()
data["County"] = data["County"].str.strip().str.title()
data["Vehicle_Use"] = data["Vehicle_Use"].str.strip().str.title()
data["Date"] = pd.to_datetime(data["Date"])
data["Year"] = data["Date"].dt.year

# ============================================================
# 5. FEATURES AND TARGET
# ============================================================
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
# 7. MODEL 1: DECISION TREE
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
plt.tight_layout()
plt.savefig("7.png", dpi=150)
plt.show()

# ============================================================
# 8. MODEL 2: KNN
# ============================================================
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

print("\n----- KNN -----")
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))
print("\nClassification Report:\n", classification_report(y_test, y_pred_knn))

# ============================================================
# 9. MODEL 3: NAIVE BAYES
# ============================================================
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)

print("\n----- NAIVE BAYES -----")
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_nb))
print("\nClassification Report:\n", classification_report(y_test, y_pred_nb))

# ============================================================
# 10. MODEL 4: LOGISTIC REGRESSION
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
plt.tight_layout()
plt.savefig("8.png", dpi=150)
plt.show()

# ============================================================
# 12. FINAL CONCLUSION
# ============================================================
best_model = results.loc[results["Accuracy"].idxmax(), "Model"]
print("\n========== FINAL CONCLUSION ==========")
print("Best model:", best_model, "with accuracy", round(results["Accuracy"].max(), 4))

print("\nRecommendations:")
print("1. Build more charging stations in top counties.")
print("2. Give incentives in counties with low EV adoption.")
print("3. Encourage EV trucks for commercial use.")
