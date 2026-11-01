import pandas as pd
import joblib

from pathlib import Path

MODEL_PATH = Path(
    "models/linear_regression_model.pkl"
)

DATA_PATH = Path("data/hourly_energy.csv")


def predict_future(hours=24):

    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["datetime"],
        index_col="datetime"
    )

    history = df["energy_kwh"].tolist()

    last_time = df.index[-1]

    forecasts = []

    for i in range(1, hours + 1):

        next_time = last_time + pd.Timedelta(hours=i)

        features = pd.DataFrame([{
            "hour": next_time.hour,
            "day_of_week": next_time.dayofweek,
            "month": next_time.month,
            "lag_1": history[-1],
            "lag_24": history[-24]
        }])

        prediction = model.predict(features)[0]

        # Consumption cannot be negative
        prediction = max(0, prediction)

        forecasts.append({
            "datetime": next_time,
            "predicted_kwh": prediction
        })

        history.append(prediction)

    return pd.DataFrame(forecasts)


if __name__ == "__main__":

    future = predict_future(24)

    print("\nNEXT 24 HOURS PREDICTION")
    print(future.to_string(index=False))

    future.to_csv(
        "outputs/future_forecast.csv",
        index=False
    )

    print("\nForecast saved successfully!")
    