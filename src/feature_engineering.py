import pandas as pd
import numpy as np

# ======================================
# LOAD DATA
# ======================================

df = pd.read_csv("data/energydata_complete.csv")

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date")

# ======================================
# TARGET (1 HOUR AHEAD)
# ======================================

# Predict 1 hour into the future
#df["target"] = df["Appliances"]
df["target"] = df["Appliances"].shift(-6)

# ======================================
# TIME FEATURES
# ======================================

df["hour"] = df["date"].dt.hour
df["day_of_week"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
df["is_weekend"] = (
    df["day_of_week"].isin([5, 6]).astype(int)
)

# Cyclical encoding
df["hour_sin"] = np.sin(
    2 * np.pi * df["hour"] / 24
)

df["hour_cos"] = np.cos(
    2 * np.pi * df["hour"] / 24
)

# ======================================
# LAG FEATURES
# ======================================

df["lag_10min"] = df["Appliances"].shift(1)
df["lag_30min"] = df["Appliances"].shift(3)
df["lag_1hour"] = df["Appliances"].shift(6)
df["lag_6hour"] = df["Appliances"].shift(36)
df["lag_24hour"] = df["Appliances"].shift(144)

# ======================================
# ROLLING FEATURES
# ======================================

df["rolling_1hour"] = (
    df["Appliances"]
    .rolling(window=6)
    .mean()
)

df["rolling_6hour"] = (
    df["Appliances"]
    .rolling(window=36)
    .mean()
)

df["rolling_24hour"] = (
    df["Appliances"]
    .rolling(window=144)
    .mean()
)

# ======================================
# DROP MISSING ROWS
# ======================================

df = df.dropna()

# ======================================
# SAVE
# ======================================
# Remove extreme spikes
upper_limit = df["target"].quantile(0.99)

df = df[df["target"] <= upper_limit]

df.to_csv(
    "data/energy_features.csv",
    index=False
)

print(df.head())
print(df.shape)
print("Feature engineering complete.")