import pandas as pd
import numpy as np

# =========================
# LOAD DATA
# =========================

df = pd.read_csv("data/energydata_complete.csv")

# Convert timestamp
df["date"] = pd.to_datetime(df["date"])

# Sort by time
df = df.sort_values("date")

# =========================
# TIME FEATURES
# =========================

# Hour of day
df["hour"] = df["date"].dt.hour

# Day of week
df["day_of_week"] = df["date"].dt.dayofweek

# Month
df["month"] = df["date"].dt.month

# Weekend flag
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

# =========================
# LAG FEATURES
# =========================

# Dataset interval = 10 minutes
# 1 hour lag = 6 rows
# 24 hour lag = 144 rows

df["lag_1_hour"] = df["Appliances"].shift(6)

df["lag_24_hour"] = df["Appliances"].shift(144)

# =========================
# ROLLING FEATURES
# =========================

# 3-hour rolling mean
df["rolling_mean_3h"] = (
    df["Appliances"]
    .rolling(window=18)
    .mean()
)

# 6-hour rolling mean
df["rolling_mean_6h"] = (
    df["Appliances"]
    .rolling(window=36)
    .mean()
)

# 24-hour rolling mean
df["rolling_mean_24h"] = (
    df["Appliances"]
    .rolling(window=144)
    .mean()
)

# =========================
# HANDLE MISSING VALUES
# =========================

print("\nMissing values BEFORE cleaning:")
print(df.isnull().sum())

# Forward fill
df = df.ffill()

# Backward fill
df = df.bfill()

print("\nMissing values AFTER cleaning:")
print(df.isnull().sum())

# =========================
# HANDLE OUTLIERS
# =========================

# Using IQR method

Q1 = df["Appliances"].quantile(0.25)
Q3 = df["Appliances"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Cap outliers
df["Appliances"] = np.clip(
    df["Appliances"],
    lower_bound,
    upper_bound
)

print("\nOutlier bounds:")
print("Lower:", lower_bound)
print("Upper:", upper_bound)

# =========================
# FINAL CHECK
# =========================

print("\nFinal dataframe shape:")
print(df.shape)

print("\nFeature columns:")
print(df.columns)

print("\nPreview:")
print(df.head())

# =========================
# SAVE FEATURED DATASET
# =========================

df.to_csv(
    "data/energy_features.csv",
    index=False
)

print("\nFeature engineered dataset saved.")