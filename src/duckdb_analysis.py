import pandas as pd
import duckdb

# Load data
df = pd.read_csv("data/energydata_complete.csv")

# Datetime conversion
df["date"] = pd.to_datetime(df["date"])

# Extract time features
df["hour"] = df["date"].dt.hour
df["day_of_week"] = df["date"].dt.day_name()
df["month"] = df["date"].dt.month_name()

# Connect DuckDB
con = duckdb.connect()

# Register dataframe
con.register("energy", df)

# Average appliance consumption by hour
hour_query = """
SELECT
    hour,
    AVG(Appliances) AS avg_consumption
FROM energy
GROUP BY hour
ORDER BY hour
"""

print("\n===== AVG BY HOUR =====")
print(con.execute(hour_query).fetchdf())

# Average by weekday
day_query = """
SELECT
    day_of_week,
    AVG(Appliances) AS avg_consumption
FROM energy
GROUP BY day_of_week
"""

print("\n===== AVG BY DAY =====")
print(con.execute(day_query).fetchdf())

# Average by month
month_query = """
SELECT
    month,
    AVG(Appliances) AS avg_consumption
FROM energy
GROUP BY month
"""

print("\n===== AVG BY MONTH =====")
print(con.execute(month_query).fetchdf())