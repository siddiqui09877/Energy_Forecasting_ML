import pandas as pd
from pathlib import Path

DATA_FILE = Path(
    "data/household_power_consumption.txt"
)

OUTPUT_FILE = Path("data/hourly_energy.csv")

print("Reading electricity dataset...")

df = pd.read_csv(
    DATA_FILE,
    sep=";",
    na_values=["?"],
    low_memory=False
)

# Combine date and time
df["datetime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"],
    dayfirst=True,
    errors="coerce"
)

# Convert power into numeric values
df["Global_active_power"] = pd.to_numeric(
    df["Global_active_power"],
    errors="coerce"
)

# Keep required columns
df = df[["datetime", "Global_active_power"]]

df = df.dropna(subset=["datetime"])
df = df.set_index("datetime").sort_index()

# Fill missing power readings
df["Global_active_power"] = (
    df["Global_active_power"]
    .interpolate(method="time")
)

# Convert minute-level average power to hourly energy
hourly = (
    df["Global_active_power"]
    .resample("h")
    .mean()
    .dropna()
    .to_frame()
)

hourly.rename(
    columns={"Global_active_power": "energy_kwh"},
    inplace=True
)

hourly.to_csv(OUTPUT_FILE)

print("Preprocessing completed!")
print(hourly.head())
print("Saved to:", OUTPUT_FILE)