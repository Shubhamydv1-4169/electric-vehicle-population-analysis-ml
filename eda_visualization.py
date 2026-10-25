import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid")

# Load and preprocess (run data_preprocessing.py first or inline)
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
# 4. EDA AND PROJECT OBJECTIVES
# ============================================================
latest_data = data[data["Date"] == data["Date"].max()]
monthly = data.groupby("Date")[["BEV", "PHEV", "EV_Total"]].sum()

# Objective 1: How has total EV grown over time?
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
plt.tight_layout()
plt.savefig("1.png", dpi=150)
plt.show()

# Objective 2: Top 10 counties with most EVs?
top_counties = latest_data.groupby("County")["EV_Total"].sum().sort_values(ascending=False).head(10)
print("\nObjective 2: Top Counties")
print(top_counties)

plt.figure(figsize=(10, 6))
bars = plt.barh(top_counties.sort_values().index, top_counties.sort_values().values, color="#4c9be8")
plt.bar_label(bars, fmt="{:,.0f}", padding=4)
plt.title("Objective 2: Top 10 Counties by EV Total", fontsize=14, fontweight="bold")
plt.xlabel("EV Total")
plt.ylabel("County")
plt.tight_layout()
plt.savefig("2.png", dpi=150)
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
plt.tight_layout()
plt.savefig("3.png", dpi=150)
plt.show()

# Objective 4: Passenger vs Truck EV distribution
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
plt.tight_layout()
plt.savefig("4.png", dpi=150)
plt.show()

# Objective 5: Percent EV change by year
year_percent = data.groupby("Year")["Percent_EV"].mean()
print("\nObjective 5: Average Percent EV by Year")
print(year_percent.round(2))

plt.figure(figsize=(10, 5))
bars = plt.bar(year_percent.index.astype(str), year_percent.values, color="#f4a261")
plt.bar_label(bars, fmt="{:.2f}%", padding=3)
plt.title("Objective 5: Average Percent EV by Year", fontsize=14, fontweight="bold")
plt.xlabel("Year")
plt.ylabel("Percent EV")
plt.tight_layout()
plt.savefig("5.png", dpi=150)
plt.show()

# Correlation Heatmap
plt.figure(figsize=(9, 7))
sns.heatmap(data[num_cols + ["Year"]].corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("6.png", dpi=150)
plt.show()

print("\nEDA complete. All plots saved.")
