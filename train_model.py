import pandas as pd
import joblib

from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

DATA_FILE = Path("data/hourly_energy.csv")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

print("Loading dataset...")

df = pd.read_csv(
    DATA_FILE,
    parse_dates=["datetime"],
    index_col="datetime"
)

# Create time-based features
df["hour"] = df.index.hour
df["day_of_week"] = df.index.dayofweek
df["month"] = df.index.month

# Create historical consumption features
df["lag_1"] = df["energy_kwh"].shift(1)
df["lag_24"] = df["energy_kwh"].shift(24)

# Predict next hour consumption
df["target"] = df["energy_kwh"].shift(-1)

df = df.dropna()

features = [
    "hour",
    "day_of_week",
    "month",
    "lag_1",
    "lag_24"
]

X = df[features]
y = df["target"]

# Chronological train-test split
split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

print("Training Linear Regression...")

model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(
    y_test, predictions
) ** 0.5

r2 = r2_score(y_test, predictions)

print("\nMODEL PERFORMANCE")
print("-------------------------")
print("MAE:", round(mae, 4), "kWh")
print("RMSE:", round(rmse, 4), "kWh")
print("R2 Score:", round(r2, 4))

# Save trained model
joblib.dump(
    model,
    MODEL_DIR / "linear_regression_model.pkl"
)

# Save actual and predicted values
results = pd.DataFrame({
    "actual": y_test,
    "predicted": predictions
}, index=y_test.index)

results.to_csv(
    OUTPUT_DIR / "test_predictions.csv"
)

# Save evaluation metrics
metrics = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "R2 Score"],
    "Value": [mae, rmse, r2]
})

metrics.to_csv(
    OUTPUT_DIR / "model_metrics.csv",
    index=False
)

print("\nModel training completed!")
print("Model saved successfully!")