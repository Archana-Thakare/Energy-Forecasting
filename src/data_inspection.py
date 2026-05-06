import pandas as pd

# Load dataset
df = pd.read_csv("data/energydata_complete.csv")

# Convert date column to datetime
df["date"] = pd.to_datetime(df["date"])

# Inspect shape
print("\n===== SHAPE =====")
print(df.shape)

# Inspect data types
print("\n===== DATA TYPES =====")
print(df.dtypes)

# Missing values
print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# Date range
print("\n===== DATE RANGE =====")
print("Start:", df["date"].min())
print("End:", df["date"].max())

# Preview
print("\n===== HEAD =====")
print(df.head())