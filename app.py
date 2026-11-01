
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Energy AI | Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM DESIGN ----------------

st.markdown("""
<style>
.stApp {
    background: #0b1220;
    color: #f1f5f9;
}
[data-testid="stSidebar"] {
    background: #111c30;
    border-right: 1px solid #263750;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}
h1, h2, h3 {
    color: #f8fafc !important;
}
p, label {
    color: #bdc9d9;
}
div[data-testid="stMetric"] {
    background: #142238;
    border: 1px solid #263d5b;
    padding: 18px;
    border-radius: 12px;
}
div[data-testid="stMetricValue"] {
    color: #38bdf8;
}
.stButton button {
    background: #087ea4;
    color: white;
    border-radius: 8px;
    border: none;
}
.stDownloadButton button {
    background: #087ea4;
    color: white;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- DATA LOADING ----------------

OUTPUT = Path("outputs")
OUTPUT.mkdir(exist_ok=True)

def load_csv(filename):
    path = OUTPUT / filename
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.warning(f"Could not read {filename}: {e}")
        return pd.DataFrame()

predictions = load_csv("test_predictions.csv")
forecast = load_csv("future_forecast.csv")
metrics = load_csv("model_metrics.csv")

# ---------------- HELPERS ----------------

def find_date_column(df):
    for col in df.columns:
        if col.lower() in ["datetime", "date", "timestamp", "time"]:
            return col

    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower():
            return col

    return None

def numeric_columns(df):
    return df.select_dtypes(include=np.number).columns.tolist()

def prepare_data(df):
    df = df.copy()
    date_col = find_date_column(df)

    if date_col:
        df[date_col] = pd.to_datetime(
            df[date_col], errors="coerce"
        )
        df = df.dropna(subset=[date_col])
        df = df.sort_values(date_col)

    return df, date_col

predictions, date_col = prepare_data(predictions)
forecast, forecast_date_col = prepare_data(forecast)

# ---------------- SIDEBAR ----------------

st.sidebar.markdown("# ⚡ ENERGY AI")
st.sidebar.caption("INTELLIGENT FORECASTING SYSTEM")
st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "Consumption Analysis",
        "Future Forecast",
        "Model Performance",
        "Data Explorer"
    ]
)

st.sidebar.divider()
st.sidebar.caption("MODEL: LINEAR REGRESSION")
st.sidebar.caption("STATUS: LOCAL DATA MODE")

# ---------------- HEADER ----------------

st.markdown(
    "<h1>⚡ Energy Intelligence Dashboard</h1>",
    unsafe_allow_html=True
)
st.caption(
    "Machine Learning Based Electricity Consumption "
    "Analysis & Prediction"
)

st.divider()

# ---------------- OVERVIEW ----------------

if page == "Overview":

    st.subheader("Energy Overview")

    nums = numeric_columns(predictions)

    actual_col = next(
        (c for c in nums if "actual" in c.lower()),
        nums[0] if nums else None
    )

    predicted_col = next(
        (c for c in nums if "predict" in c.lower()),
        nums[1] if len(nums) > 1 else None
    )

    if actual_col:
        actual = predictions[actual_col].dropna()
        avg = actual.mean()
        peak = actual.max()
        total = actual.sum()
    else:
        avg = peak = total = 0

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Average Consumption", f"{avg:.3f}")
    c2.metric("Peak Consumption", f"{peak:.3f}")
    c3.metric("Total Recorded", f"{total:.2f}")
    c4.metric("Data Records", f"{len(predictions):,}")

    st.divider()

    st.subheader("Consumption Trend")

    if actual_col and date_col:
        fig = px.line(
            predictions,
            x=date_col,
            y=actual_col,
            title="Historical Electricity Consumption",
            template="plotly_dark"
        )
        fig.update_traces(line=dict(color="#38bdf8", width=2))
        fig.update_layout(
            paper_bgcolor="#0b1220",
            plot_bgcolor="#111c30",
            height=430
        )
        st.plotly_chart(fig, use_container_width=True)

    elif actual_col:
        st.line_chart(predictions[actual_col])

    else:
        st.warning("No prediction data found. Run train_model.py first.")

    st.subheader("Future Forecast Preview")

    if not forecast.empty:
        st.dataframe(forecast.head(10), use_container_width=True)
    else:
        st.info("Run forecast.py to generate future predictions.")

# ---------------- CONSUMPTION ANALYSIS ----------------

elif page == "Consumption Analysis":

    st.subheader("Consumption Analysis")

    if predictions.empty:
        st.warning("No test_predictions.csv found.")
    else:
        nums = numeric_columns(predictions)
        if not nums:
            st.warning("No numeric consumption columns found.")
        else:
            selected = st.selectbox(
                "Select consumption measurement",
                nums
            )

            if date_col:
                min_date = predictions[date_col].min().date()
                max_date = predictions[date_col].max().date()

                date_range = st.date_input(
                    "Select date range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )

                filtered = predictions.copy()

                if isinstance(date_range, tuple) and len(date_range) == 2:
                    filtered = filtered[
                        (filtered[date_col].dt.date >= date_range[0]) &
                        (filtered[date_col].dt.date <= date_range[1])
                    ]
            else:
                filtered = predictions.copy()

            if not filtered.empty:
                fig = px.line(
                    filtered,
                    x=date_col if date_col else filtered.index,
                    y=selected,
                    title="Selected Consumption Trend",
                    template="plotly_dark"
                )
                fig.update_layout(
                    paper_bgcolor="#0b1220",
                    plot_bgcolor="#111c30",
                    height=450
                )
                st.plotly_chart(fig, use_container_width=True)

                st.metric(
                    "Selected Average",
                    f"{filtered[selected].mean():.3f}"
                )

# ---------------- FUTURE FORECAST ----------------

elif page == "Future Forecast":

    st.subheader("Future Energy Prediction")

    if forecast.empty:
        st.warning("Future forecast file not found. Run forecast.py first.")
    else:
        st.success("Future forecast data loaded.")

        nums = numeric_columns(forecast)

        if nums:
            selected = st.selectbox(
                "Select forecast measurement",
                nums
            )

            records = st.slider(
                "Number of forecast records to display",
                min_value=1,
                max_value=max(1, len(forecast)),
                value=min(24, len(forecast))
            )

            view = forecast.head(records)

            fig = px.line(
                view,
                x=forecast_date_col if forecast_date_col else view.index,
                y=selected,
                title="Future Electricity Consumption Prediction",
                template="plotly_dark",
                markers=True
            )

            fig.update_layout(
                paper_bgcolor="#0b1220",
                plot_bgcolor="#111c30",
                height=450
            )

            st.plotly_chart(fig, use_container_width=True)

            st.metric(
                "Average Forecast",
                f"{view[selected].mean():.3f}"
            )

            st.download_button(
                "Download Future Forecast CSV",
                forecast.to_csv(index=False),
                "energy_forecast.csv",
                "text/csv"
            )
        else:
            st.warning("No numeric forecast columns found.")

# ---------------- MODEL PERFORMANCE ----------------

elif page == "Model Performance":

    st.subheader("Linear Regression Performance")

    if metrics.empty:
        st.warning("model_metrics.csv not found.")
    else:
        st.dataframe(metrics, use_container_width=True)

        metric_values = metrics.select_dtypes(
            include=np.number
        )

        if not metric_values.empty:
            cols = st.columns(len(metric_values.columns))

            for i, col in enumerate(metric_values.columns):
                value = metric_values[col].iloc[0]
                cols[i].metric(col.upper(), f"{value:.4f}")

    nums = numeric_columns(predictions)

    actual_col = next(
        (c for c in nums if "actual" in c.lower()),
        None
    )
    predicted_col = next(
        (c for c in nums if "predict" in c.lower()),
        None
    )

    if actual_col and predicted_col:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            y=predictions[actual_col],
            name="Actual",
            mode="lines",
            line=dict(color="#38bdf8")
        ))

        fig.add_trace(go.Scatter(
            y=predictions[predicted_col],
            name="Predicted",
            mode="lines",
            line=dict(color="#f59e0b")
        ))

        fig.update_layout(
            title="Actual vs Predicted Consumption",
            template="plotly_dark",
            paper_bgcolor="#0b1220",
            plot_bgcolor="#111c30",
            height=450
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info(
            "Actual and predicted columns were not automatically "
            "identified. Check the Data Explorer."
        )

# ---------------- DATA EXPLORER ----------------

elif page == "Data Explorer":

    st.subheader("Dataset Explorer")

    choice = st.radio(
        "Choose dataset",
        ["Test Predictions", "Future Forecast"],
        horizontal=True
    )

    df = predictions if choice == "Test Predictions" else forecast

    if not df.empty:
        st.write("Dataset shape:", df.shape)
        st.dataframe(df, use_container_width=True)

        st.download_button(
            "Download Dataset CSV",
            df.to_csv(index=False),
            "energy_data.csv",
            "text/csv"
        )
    else:
        st.warning("Selected dataset is empty or missing.")

st.divider()
st.caption(
    "ENERGY AI | Linear Regression | Machine Learning Project"
)
