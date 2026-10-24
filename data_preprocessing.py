import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid")

# ============================================================
# 1. DATASET LOADING AND INFORMATION
# ============================================================
data = pd.read_csv("Electric_Vehicle_Population_Size_History_By_County_.csv")
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
num_cols = ["BEV", "PHEV", "EV_Total", "Non_EV_Total", "Total_Vehicles", "Percent_EV"]
data[num_cols] = data[num_cols].replace("[,%]", "", regex=True).apply(pd.to_numeric, errors="coerce")

# Missing values
print("\nMissing Values:")
print(data.isna().sum())
data = data.dropna()

# Duplicates
print("\nDuplicate rows:", data.duplicated().sum())
data = data.drop_duplicates()

# Inconsistent data (extra spaces and different casing)
data["County"] = data["County"].str.strip().str.title()
data["Vehicle_Use"] = data["Vehicle_Use"].str.strip().str.title()

# Convert Date to datetime
data["Date"] = pd.to_datetime(data["Date"])
print("\nRows after cleaning:", data.shape[0])

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
data["Year"] = data["Date"].dt.year

print("\nPreprocessing complete. Data shape:", data.shape)
print("\nColumns:", list(data.columns))
print("\nDate range:", data["Date"].min(), "to", data["Date"].max())
