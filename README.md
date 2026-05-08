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

### Hourly Energy Consumption Breakdown using SQL

| Time Window | Consumption (Units) | Category | Description |
| :--- | :--- | :--- | :--- |
| **00:00 - 04:00** | 48 – 52 | **Baseload** | Always-on appliances (refrigerators, standby electronics). |
| **06:00 - 08:00** | 57.7 → 106.1 | **Morning Rise** | Initial spike due to morning routines and breakfast. |
| **11:00** | 133.13 | **Secondary Peak** | Mid-day activity or small commercial operations. |
| **18:00** | 190.36 | **Primary Peak** | Evening surge (HVAC, cooking, and entertainment). |
