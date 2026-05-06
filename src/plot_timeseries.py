import pandas as pd
import plotly.express as px
import plotly.io as pio

# Force browser rendering
pio.renderers.default = "browser"

# Load data
df = pd.read_csv("data/energydata_complete.csv")

# Convert datetime
df["date"] = pd.to_datetime(df["date"])

# Create chart
fig = px.line(
    df,
    x="date",
    y="Appliances",
    title="Appliance Energy Consumption Over Time"
)

# Show chart
fig.show()