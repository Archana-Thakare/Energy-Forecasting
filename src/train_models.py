import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ======================================
# LOAD DATA
# ======================================

df = pd.read_csv("data/energy_features.csv")

# ======================================
# FEATURES + TARGET
# ======================================

target = "target"

features_to_drop = [
    "date",
    "target"
]

selected_features = [

    # Time features
    "hour",
    "day_of_week",
    "is_weekend",
    "hour_sin",
    "hour_cos",

    # Lag features
    "lag_10min",
    "lag_30min",
    "lag_1hour",
    "lag_6hour",
    "lag_24hour",

    # Rolling features
    "rolling_1hour",
    "rolling_6hour",
    "rolling_24hour"
]

X = df[selected_features]

y = df[target]

# ======================================
# TIME SPLIT
# ======================================

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

# ======================================
# EVALUATION FUNCTION
# ======================================

def evaluate(model, X_test, y_test):

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    return mae, rmse, r2

# ======================================
# LINEAR REGRESSION
# ======================================

lr = LinearRegression()

lr.fit(X_train, y_train)

mae, rmse, r2 = evaluate(
    lr,
    X_test,
    y_test
)

print("\nLinear Regression")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)

# ======================================
# RANDOM FOREST
# ======================================

rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=6,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

mae, rmse, r2 = evaluate(
    rf,
    X_test,
    y_test
)

print("\nRandom Forest")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)

# ======================================
# XGBOOST
# ======================================

xgb = XGBRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1,
    reg_lambda=1,
    objective="reg:squarederror",
    random_state=42
)

xgb.fit(X_train, y_train)

mae, rmse, r2 = evaluate(
    xgb,
    X_test,
    y_test
)

print("\nXGBoost")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)