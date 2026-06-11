import pandas as pd
import numpy as np
import joblib

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

mlflow.set_tracking_uri("file:./mlruns")

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
    "rolling_24hour",

    # Temperature & Humidity features
    ##"T1",
    ##"RH_1",
    #"T2",
    #"RH_2",
    #"T3",
    ##"RH_3",
    #"T4",
    #"RH_4",
    #"T5",
    #"RH_5",
    #"T6",
    #"RH_6",
    #"T7",
    #"RH_7",
    #"T8",
    #"RH_8",
    #"T9",
    #"RH_9"
    #"T_out",
    #"Press_mm_hg",
    ##"RH_out",
    #"Windspeed",
    ##"Visibility",
    ##"Tdewpoint",
    ##"rv1"
    #"rv2"

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

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    return mae, rmse, r2


# =====================================
# MLFLOW EXPERIMENT
# =====================================

mlflow.set_experiment("Energy Forecasting")


# ======================================
# LINEAR REGRESSION
# ======================================

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

# ======================================
# RANDOM FOREST
# ======================================

with mlflow.start_run(run_name="Random Forest"):
    rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=6,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
    )

    rf.fit(X_train, y_train)

    mae, rmse, r2 = evaluate_model(
    rf,
    X_test,
    y_test
    )
    # Log parameters
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 6)

    # Log metrics
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2)

    # Log model
    mlflow.sklearn.log_model(
        rf,
        "random_forest_model"
    )

    print("\nRandom Forest")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2:", r2)

# ======================================
# XGBOOST
# ======================================
with mlflow.start_run(run_name="XGBoost"):
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

    mae, rmse, r2 = evaluate_model(
        xgb,
        X_test,
        y_test
    )

    joblib.dump(xgb, "models/xgboost_model.pkl")
    
    # Log parameters
    mlflow.log_param("model_type", "XGBoost")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("learning_rate", 0.05)
    mlflow.log_param("max_depth", 4)

    # Log metrics
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2)

    # Log model
    mlflow.xgboost.log_model(
        xgb,
        "xgboost_model"
    )
    print("\nXGBoost")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2:", r2)