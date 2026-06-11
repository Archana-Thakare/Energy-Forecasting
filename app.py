import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# ======================================
# PAGE CONFIG
# ======================================

st.set_page_config(
    page_title="Energy Forecast Dashboard",
    layout="wide"
)

st.title("Energy Consumption Forecasting Dashboard")

# ======================================
# LOAD DATA
# ======================================

df = pd.read_csv("data/energy_features.csv")

df["date"] = pd.to_datetime(df["date"])

# ======================================
# LOAD MODEL
# ======================================

model = joblib.load("models/xgboost_model.pkl")

# ======================================
# FEATURES
# ======================================

target = "Appliances"

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
    #"T1",
    #"RH_1",
    #"T2",
    #"RH_2",
    #"T3",
    #"RH_3",
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
    #"RH_out",
    #"Windspeed",
    #"Visibility",
    #"Tdewpoint",
    #"rv1"
    #"rv2"


]

X = df[selected_features]


# ======================================
# PREDICTIONS
# ======================================

df["prediction"] = model.predict(X)

# ======================================
# DATE FILTER
# ======================================

st.sidebar.header("Filter Data")

start_date = st.sidebar.date_input(
    "Start Date",
    df["date"].min().date()
)

end_date = st.sidebar.date_input(
    "End Date",
    df["date"].max().date()
)

# Filter dataframe
filtered_df = df[
    (df["date"].dt.date >= start_date) &
    (df["date"].dt.date <= end_date)
]

# ======================================
# ACTUAL VS PREDICTED
# ======================================

st.subheader("Actual vs Predicted Consumption")

fig = px.line(
    filtered_df,
    x="date",
    y=["Appliances", "prediction"],
    labels={
        "value": "Energy Consumption",
        "date": "Date"
    }
)

st.plotly_chart(fig, use_container_width=True)

# ======================================
# METRICS
# ======================================

st.subheader("Summary Statistics")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Average Actual Consumption",
        round(filtered_df["Appliances"].mean(), 2)
    )

with col2:
    st.metric(
        "Average Predicted Consumption",
        round(filtered_df["prediction"].mean(), 2)
    )

# ======================================
# FEATURE IMPORTANCE
# ======================================

st.subheader("Feature Importance")

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

fig_importance = px.bar(
    importance_df.head(15),
    x="Importance",
    y="Feature",
    orientation="h",
    title="Top 15 Important Features"
)

st.plotly_chart(
    fig_importance,
    use_container_width=True
)

# ======================================
# DATA PREVIEW
# ======================================

st.subheader("Filtered Data Preview")

st.dataframe(
    filtered_df.head(20)
)

# ======================================
# DOWNLOAD OPTION
# ======================================

csv = filtered_df.to_csv(index=False)

st.download_button(
    label="Download Predictions CSV",
    data=csv,
    file_name="energy_predictions.csv",
    mime="text/csv"
)