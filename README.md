# Energy-Forecasting
Time-series machine learning pipeline that forecasts household energy consumption.

The Dataset
Uses the UCI Appliances Energy Prediction dataset https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction. It contains energy consumption readings from a
low-energy house, measured every 10 minutes, along with temperature and humidity sensors across
multiple rooms.
Download here: UCI Appliances Energy Prediction Dataset
The dataset includes: timestamps, appliance energy use (Wh), lights energy use (Wh), temperature
and humidity across 9 rooms, outdoor weather data (temperature, humidity, wind speed, visibility, dew
point).

### Exploratory Data Analysis
The dataset was analyzed using Pandas, DuckDB and Plotly  

SQL queries with DuckDB were used to calculate average consumption by hour, average consumption by weekday, monthly trends.  

#### Hourly Energy Consumption Breakdown using SQL

| Time Window | Consumption (Units) | Category | Description |
| :--- | :--- | :--- | :--- |
| **00:00 - 04:00** | 48 – 52 | **Baseload** | Always-on appliances (refrigerators, standby electronics). |
| **06:00 - 08:00** | 57.7 → 106.1 | **Morning Rise** | Initial spike due to morning routines and breakfast. |
| **11:00** | 133.13 | **Secondary Peak** | Mid-day activity or small commercial operations. |
| **18:00** | 190.36 | **Primary Peak** | Evening surge (HVAC, cooking, and entertainment). |

**Weekly Anomalies**: Monday represents the **highest usage day (111.45)**, showing a sharp increase compared to Tuesday (87.12). This suggests a weekly "catch-up" period for chores or specific operational start-up routines.  
**Seasonal Stability**: Monthly consumption is exceptionally **consistent**, maintaining a narrow range of 94–100 units. This indicates that energy draw is tied to year-round appliances or fixed processes rather than seasonal heating or cooling.  

### Feature Engineering ###

Several time-series forecasting features were created Eg. hour of day, day of week, month, weekend indicator, Lag Features  

Historical consumption patterns were captured using 1-hour lag, 24-hour lag  

Rolling averages were added to smooth short-term fluctuations: 3-hour rolling mean, 6-hour rolling mean
24-hour rolling mean.

**Data Cleaning**
Missing values handled using forward/backward fill.  
Outliers capped using IQR-based clipping.  

### Machine Learning Models ###
1. Linear Regression - Simple baseline model.  
2. Random Forest Regressor - Captured non-linear relationships and feature interactions.  
3. XGBoost Regressor - Provided the strongest predictive performance due to gradient boosting, handling non-linearity and robustness to noisy tabular data

### Tech Stack ###
Programming Language - Python, IDE - Visual Studio Code, Data Processing -	Pandas, SQL Analytics - DuckDB, Machine Learning - scikit-learn, Gradient Boosting - XGBoost, Experiment Tracking -	MLflow, 
Dashboarding -	Streamlit, Visualization - Plotly, Model Serialization - joblib, Version Control - GitHub.  

### Results & KPIs ###
Evaluation Metrics :RMSE, MAE, R².  
**Model Comparison**
Linear Regression -	Weak baseline performance  
Random Forest -	Improved non-linear prediction  
XGBoost	Best overall accuracy

### What I Learned ###
1. Time-Series Feature Engineering Is Critical - Lag variables and rolling averages significantly improved forecasting accuracy.

2. Simpler Models Are Useful Baselines - Linear Regression established a benchmark for evaluating more advanced models.

3. Tree-Based Models Handle Real-World Data Better - Random Forest and XGBoost handled noise, non-linearity, outliers more effectively than linear models.

4. MLflow Simplifies Experiment Tracking : Made it easy to compare experiments, store parameters
track metrics, save trained models

5. Streamlit Enables Rapid Deployment - Allowed the project to be converted into an interactive dashboard quickly with minimal frontend code.

### What Surprised Me ###
Lag features had extremely high predictive power. Daily seasonality was stronger than monthly seasonality. XGBoost handled noisy consumption spikes surprisingly well. Even relatively simple engineered features produced strong forecasting results.  

### Data Improvements ###
Add external weather APIs, Real-time prediction updates, MLOps Enhancements, Automated retraining pipeline, Docker deployment, Cloud deployment using AWS/Azure/GCP, Add streaming ingestion pipelines

### Conclusion ###
This project successfully built an end-to-end machine learning pipeline for energy consumption forecasting, covering data ingestion, feature engineering, exploratory analysis, predictive modeling, experiment tracking, interactive dashboard deployment.  

Among all tested models, XGBoost delivered the best forecasting performance and demonstrated the effectiveness of feature-engineered machine learning approaches for energy prediction tasks.