import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "data/hourly_energy.csv",
    parse_dates=["datetime"],
    index_col="datetime"
)

print("\nDataset Information")
print(df.info())

print("\nStatistical Analysis")
print(df.describe())

# Energy consumption graph
plt.figure(figsize=(12, 5))
plt.plot(df.index[:500], df["energy_kwh"][:500])

plt.title("Hourly Energy Consumption")
plt.xlabel("Date")
plt.ylabel("Energy (kWh)")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("outputs/energy_consumption.png")
plt.show()

# Distribution graph
plt.figure(figsize=(8, 5))
plt.hist(df["energy_kwh"], bins=30)

plt.title("Energy Consumption Distribution")
plt.xlabel("Energy (kWh)")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig("outputs/energy_distribution.png")
plt.show()

print("EDA completed successfully!")