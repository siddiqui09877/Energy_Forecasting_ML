
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Energy AI | Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# PROJECT FILE PATHS
# All CSV files are currently in the GitHub root folder
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

PREDICTIONS_FILE = BASE_DIR / "test_predictions.csv"
FORECAST_FILE = BASE_DIR / "future_forecast.csv"
METRICS_FILE = BASE_DIR / "model_metrics.csv"
DATA_FILE = BASE_DIR / "hourly_energy.csv"

# ==================================================
# CUSTOM DASHBOARD DESIGN
# ==================================================

st.markdown("""
<style>

.stApp {
    background-color: #0b1220;
    color: #f1f5f9;
}

[data-testid="stSidebar"] {
    background-color: #111c30;
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
    background-color: #142238;
    border: 1px solid #263d5b;
    padding: 18px;
    border-radius: 12px;
}

div[data-testid="stMetricValue"] {
    color: #38bdf8;
}

.stButton button,
.stDownloadButton button {
    background-color: #087ea4;
    color: white;
    border-radius: 8px;
    border: none;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# DATA LOADING
# ==================================================

@st.cache_data
def load_csv(file_path):
    if not file_path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(file_path)
    except Exception as error:
        st.error(f"Error loading {file_path.name}: {error}")
        return pd.DataFrame()


predictions = load_csv(PREDICTIONS_FILE)
forecast = load_csv(FORECAST_FILE)
metrics = load_csv(METRICS_FILE)
original_data = load_csv(DATA_FILE)

# ==================================================
# HELPER FUNCTIONS
# ==================================================

def find_date_column(df):
    for col in df.columns:
        name = col.lower()

        if name in ["datetime", "date", "timestamp", "time"]:
            return col

    for col in df.columns:
        name = col.lower()

        if "date" in name or "time" in name:
            return col

    return None


def numeric_columns(df):
    return df.select_dtypes(
        include=np.number
    ).columns.tolist()


def prepare_data(df):
    df = df.copy()

    date_col = find_date_column(df)

    if date_col:
        df[date_col] = pd.to_datetime(
            df[date_col],
            errors="coerce"
        )

        df = df.dropna(
            subset=[date_col]
        )

        df = df.sort_values(
            date_col
        )

    return df, date_col


predictions, date_col = prepare_data(predictions)
forecast, forecast_date_col = prepare_data(forecast)

# ==================================================
# SIDEBAR NAVIGATION
# ==================================================

st.sidebar.markdown("# ⚡ ENERGY AI")

st.sidebar.caption(
    "INTELLIGENT FORECASTING SYSTEM"
)

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

st.sidebar.caption(
    "MODEL: LINEAR REGRESSION"
)

st.sidebar.caption(
    "STATUS: DATA LOADED FROM PROJECT FILES"
)

# ==================================================
# MAIN HEADER
# ==================================================

st.markdown(
    "<h1>⚡ Energy Intelligence Dashboard</h1>",
    unsafe_allow_html=True
)

st.caption(
    "Machine Learning Based Electricity Consumption "
    "Analysis and Prediction"
)

st.divider()

# ==================================================
# PAGE 1: OVERVIEW
# ==================================================

if page == "Overview":

    st.subheader("Energy Overview")

    nums = numeric_columns(predictions)

    actual_col = next(
        (
            col for col in nums
            if "actual" in col.lower()
        ),
        nums[0] if nums else None
    )

    if actual_col:

        actual = predictions[actual_col].dropna()

        average = actual.mean()
        peak = actual.max()
        total = actual.sum()

    else:

        average = 0
        peak = 0
        total = 0

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Average Consumption",
        f"{average:.3f}"
    )

    c2.metric(
        "Peak Consumption",
        f"{peak:.3f}"
    )

    c3.metric(
        "Total Recorded",
        f"{total:.2f}"
    )

    c4.metric(
        "Data Records",
        f"{len(predictions):,}"
    )

    st.divider()

    st.subheader("Historical Consumption Trend")

    if not predictions.empty and actual_col:

        fig = px.line(
            predictions,
            x=date_col if date_col else predictions.index,
            y=actual_col,
            title="Historical Electricity Consumption",
            template="plotly_dark"
        )

        fig.update_traces(
            line=dict(
                color="#38bdf8",
                width=2
            )
        )

        fig.update_layout(
            paper_bgcolor="#0b1220",
            plot_bgcolor="#111c30",
            height=430
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.warning(
            "Prediction data is unavailable. "
            "Check test_predictions.csv."
        )

    st.subheader("Future Forecast Preview")

    if not forecast.empty:

        st.dataframe(
            forecast.head(10),
            use_container_width=True
        )

    else:

        st.info(
            "Future forecast file is not available."
        )

# ==================================================
# PAGE 2: CONSUMPTION ANALYSIS
# ==================================================

elif page == "Consumption Analysis":

    st.subheader("Electricity Consumption Analysis")

    if predictions.empty:

        st.warning(
            "test_predictions.csv is missing or empty."
        )

    else:

        nums = numeric_columns(predictions)

        if not nums:

            st.warning(
                "No numeric consumption columns found."
            )

        else:

            selected = st.selectbox(
                "Select consumption measurement",
                nums
            )

            filtered = predictions.copy()

            if date_col:

                min_date = predictions[date_col].min().date()
                max_date = predictions[date_col].max().date()

                date_range = st.date_input(
                    "Select date range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )

                if (
                    isinstance(date_range, tuple)
                    and len(date_range) == 2
                ):

                    filtered = predictions[
                        (
                            predictions[date_col].dt.date
                            >= date_range[0]
                        )
                        &
                        (
                            predictions[date_col].dt.date
                            <= date_range[1]
                        )
                    ]

            if not filtered.empty:

                fig = px.line(
                    filtered,
                    x=(
                        date_col
                        if date_col
                        else filtered.index
                    ),
                    y=selected,
                    title="Electricity Consumption Trend",
                    template="plotly_dark"
                )

                fig.update_layout(
                    paper_bgcolor="#0b1220",
                    plot_bgcolor="#111c30",
                    height=450
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                st.metric(
                    "Average Consumption",
                    f"{filtered[selected].mean():.3f}"
                )

                st.metric(
                    "Maximum Consumption",
                    f"{filtered[selected].max():.3f}"
                )

            else:

                st.warning(
                    "No records available for the selected dates."
                )

# ==================================================
# PAGE 3: FUTURE FORECAST
# ==================================================

elif page == "Future Forecast":

    st.subheader("Future Energy Consumption Prediction")

    if forecast.empty:

        st.warning(
            "future_forecast.csv is missing or empty."
        )

    else:

        st.success(
            "Future forecast data loaded successfully."
        )

        nums = numeric_columns(forecast)

        if nums:

            selected = st.selectbox(
                "Select forecast measurement",
                nums
            )

            records = st.slider(
                "Number of forecast records",
                min_value=1,
                max_value=len(forecast),
                value=min(24, len(forecast))
            )

            view = forecast.head(records)

            fig = px.line(
                view,
                x=(
                    forecast_date_col
                    if forecast_date_col
                    else view.index
                ),
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

            st.plotly_chart(
                fig,
                use_container_width=True
            )

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

            st.warning(
                "No numeric forecast columns found."
            )

# ==================================================
# PAGE 4: MODEL PERFORMANCE
# ==================================================

elif page == "Model Performance":

    st.subheader("Linear Regression Model Performance")

    if metrics.empty:

        st.warning(
            "model_metrics.csv is missing or empty."
        )

    else:

        st.write("Actual model evaluation results:")

        st.dataframe(
            metrics,
            use_container_width=True
        )

        metric_values = metrics.select_dtypes(
            include=np.number
        )

        if not metric_values.empty:

            cols = st.columns(
                len(metric_values.columns)
            )

            for i, col in enumerate(
                metric_values.columns
            ):

                value = metric_values[col].iloc[0]

                if pd.notna(value):

                    cols[i].metric(
                        col.upper(),
                        f"{value:.4f}"
                    )

    nums = numeric_columns(predictions)

    actual_col = next(
        (
            col for col in nums
            if "actual" in col.lower()
        ),
        None
    )

    predicted_col = next(
        (
            col for col in nums
            if "predict" in col.lower()
        ),
        None
    )

    if (
        not predictions.empty
        and actual_col
        and predicted_col
    ):

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                y=predictions[actual_col],
                name="Actual",
                mode="lines",
                line=dict(color="#38bdf8")
            )
        )

        fig.add_trace(
            go.Scatter(
                y=predictions[predicted_col],
                name="Predicted",
                mode="lines",
                line=dict(color="#f59e0b")
            )
        )

        fig.update_layout(
            title="Actual vs Predicted Consumption",
            template="plotly_dark",
            paper_bgcolor="#0b1220",
            plot_bgcolor="#111c30",
            height=450
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.info(
            "Actual and predicted columns were not "
            "automatically identified in test_predictions.csv."
        )

# ==================================================
# PAGE 5: DATA EXPLORER
# ==================================================

elif page == "Data Explorer":

    st.subheader("Dataset Explorer")

    datasets = {
        "Test Predictions": predictions,
        "Future Forecast": forecast,
        "Original Energy Dataset": original_data
    }

    available = [
        name for name, df in datasets.items()
        if not df.empty
    ]

    if not available:

        st.warning(
            "No dataset files are available."
        )

    else:

        choice = st.radio(
            "Choose dataset",
            available,
            horizontal=True
        )

        df = datasets[choice]

        st.write(
            "Dataset shape:",
            df.shape
        )

        st.dataframe(
            df,
            use_container_width=True
        )

        st.download_button(
            "Download Dataset CSV",
            df.to_csv(index=False),
            "energy_data.csv",
            "text/csv"
        )

# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "ENERGY AI | Linear Regression | "
    "Machine Learning Based Energy Forecasting Project"
)
