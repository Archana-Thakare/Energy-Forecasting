import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import joblib

# =====================================
# LOAD FEATURE ENGINEERED DATA
# =====================================

df = pd.read_csv("data/energy_features.csv")

# Remove datetime column
df = df.drop(columns=["date"])

# =====================================
# DEFINE FEATURES + TARGET
# =====================================

target = "Appliances"

X = df.drop(columns=[target])
y = df[target]

# =====================================
# TIME-BASED TRAIN TEST SPLIT
# =====================================

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Train Shape:", X_train.shape)
print("Test Shape:", X_test.shape)

# =====================================
# METRICS FUNCTION
# =====================================

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    r2 = r2_score(y_test, predictions)

    return mae, rmse, r2

# =====================================
# MLFLOW EXPERIMENT
# =====================================

mlflow.set_experiment("Energy Forecasting")

# =====================================
# 1. LINEAR REGRESSION
# =====================================

with mlflow.start_run(run_name="Linear Regression"):

    lr_model = LinearRegression()

    lr_model.fit(X_train, y_train)

    mae, rmse, r2 = evaluate_model(
        lr_model,
        X_test,
        y_test
    )

    # Log parameters
    mlflow.log_param("model_type", "LinearRegression")

    # Log metrics
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2)

    # Log model
    mlflow.sklearn.log_model(
        lr_model,
        "linear_regression_model"
    )

    print("\nLinear Regression")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2:", r2)

# =====================================
# 2. RANDOM FOREST
# =====================================

with mlflow.start_run(run_name="Random Forest"):

    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    rf_model.fit(X_train, y_train)

    mae, rmse, r2 = evaluate_model(
        rf_model,
        X_test,
        y_test
    )

    # Log parameters
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)

    # Log metrics
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2)

    # Log model
    mlflow.sklearn.log_model(
        rf_model,
        "random_forest_model"
    )

    print("\nRandom Forest")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2:", r2)

# =====================================
# 3. XGBOOST
# =====================================

with mlflow.start_run(run_name="XGBoost"):

    xgb_model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )

    xgb_model.fit(X_train, y_train)

    mae, rmse, r2 = evaluate_model(
        xgb_model,
        X_test,
        y_test
    )

    # Log parameters
    mlflow.log_param("model_type", "XGBoost")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_param("max_depth", 6)

    # Log metrics
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2)

    # Log model
    mlflow.xgboost.log_model(
        xgb_model,
        "xgboost_model"
    )

    print("\nXGBoost")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2:", r2)

print("\nTraining complete.")
joblib.dump(xgb_model, "models/xgboost_model.pkl")