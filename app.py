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

X = df.drop(columns=[target, "date"])

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