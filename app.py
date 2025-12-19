import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from datetime import time

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Delhi Traffic Flow Prediction",
    page_icon="🚦",
    layout="centered"
)

# --------------------------------------------------
# Helper Functions
# --------------------------------------------------
def traffic_message(value):
    if value < 0.33:
        return "🟢 Low Traffic", "Smooth traffic conditions. Ideal time to travel."
    elif value < 0.66:
        return "🟡 Moderate Traffic", "Average traffic. Expect minor delays."
    else:
        return "🔴 High Traffic", "Heavy traffic expected. Consider alternate routes."

def is_peak_hour(hour):
    return (8 <= hour <= 10) or (18 <= hour <= 21)

# --------------------------------------------------
# Load Model, Scaler & Metadata
# --------------------------------------------------
# Load model
model = load_model("models/lstm_traffic_model.keras", compile=False)

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load metadata
with open("metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

SEQ_LEN = metadata["sequence_length"]
# Force feature order from scaler (single source of truth)
FEATURE_COLS = list(scaler.feature_names_in_)


# --------------------------------------------------
# Load Processed Traffic Data
# --------------------------------------------------
try:
    hourly_df = pd.read_csv(
        "data/hourly_processed.csv",
        parse_dates=["timestamp"]
    )
except FileNotFoundError:
    st.error("❌ Processed data file not found.")
    st.stop()

hourly_df["date"] = hourly_df["timestamp"].dt.date

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown(
    """
    <h1 style='text-align:center;'>🚦 Delhi Traffic Flow Prediction</h1>
    <p style='text-align:center; font-size:18px;'>
    Predict traffic density in advance and plan smarter travel.
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# User Input
# --------------------------------------------------
st.subheader("📥 Enter Prediction Details")

available_dates = sorted(hourly_df["date"].unique())

date_input = st.date_input(
    "Select Date",
    min_value=available_dates[0],
    max_value=available_dates[-1],
    value=available_dates[-1]
)

hour_input = st.slider(
    "Select Hour",
    min_value=0,
    max_value=23,
    value=9
)

predict_btn = st.button("🚀 Predict Traffic")

# --------------------------------------------------
# Prediction Logic
# --------------------------------------------------
if predict_btn:

    daily_data = hourly_df[hourly_df["date"] == date_input]

    if daily_data.empty:
        st.error("No data available for selected date.")
        st.stop()

    st.markdown("## 📈 Traffic Trend (Selected Date)")
    st.line_chart(daily_data.set_index("hour")["avg_queue_density"])

    cutoff_time = pd.Timestamp.combine(date_input, time(hour_input, 0))
    last_seq = hourly_df[hourly_df["timestamp"] < cutoff_time].tail(SEQ_LEN)

    if len(last_seq) < SEQ_LEN:
        st.error("Not enough historical data for prediction.")
        st.stop()

    if not set(FEATURE_COLS).issubset(last_seq.columns):
        st.error("Feature mismatch between training and app data.")
        st.stop()
    # --------------------------------------------------
    # Create cyclical features (MUST match training)
    # --------------------------------------------------
    last_seq = last_seq.copy()

    last_seq["hour_sin"] = np.sin(2 * np.pi * last_seq["hour"] / 24)
    last_seq["hour_cos"] = np.cos(2 * np.pi * last_seq["hour"] / 24)

    last_seq["dow_sin"] = np.sin(2 * np.pi * last_seq["day_of_week"] / 7)
    last_seq["dow_cos"] = np.cos(2 * np.pi * last_seq["day_of_week"] / 7)

    # Select ONLY the features used during training
    model_features = last_seq[FEATURE_COLS]

    scaled_input = scaler.transform(model_features)
    model_input = scaled_input.reshape(1, SEQ_LEN, len(FEATURE_COLS))


    prediction = model.predict(model_input)[0][0]
    prediction = float(np.clip(prediction, 0, 1))

    level, message = traffic_message(prediction)
    predicted_hour = (hour_input + 1) % 24

    # --------------------------------------------------
    # Results
    # --------------------------------------------------
    st.markdown("## 🚦 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Predicted Queue Density (Next Hour)",
            f"{prediction:.3f}"
        )
        st.caption("Normalized value (0–1)")

    with col2:
        if "Low" in level:
            st.success(level)
        elif "Moderate" in level:
            st.warning(level)
        else:
            st.error(level)

    st.info(message)

    if is_peak_hour(predicted_hour):
        st.warning(
            f"⏰ Peak Hour Alert ({predicted_hour}:00 hrs)\n\n"
            "Traffic congestion is usually higher during this time."
        )

    st.caption(f"🕒 Prediction for {predicted_hour}:00 hrs")

    # --------------------------------------------------
    # Heatmaps
    # --------------------------------------------------
    st.markdown("## 🔥 Hour-wise Traffic Heatmap (Selected Date)")

    heatmap_day = (
        daily_data
        .set_index("hour")[["avg_queue_density"]]
        .T
    )

    st.dataframe(
        heatmap_day.style
        .background_gradient(cmap="RdYlGn", axis=1)
        .format("{:.3f}"),
        use_container_width=True
    )

    st.markdown("## 🔥 Traffic Heatmap (Last 7 Days × Hour)")

    last_7_days = hourly_df[hourly_df["date"] <= date_input].tail(7 * 24)

    heatmap_week = last_7_days.pivot_table(
        values="avg_queue_density",
        index="date",
        columns="hour"
    )

    st.dataframe(
        heatmap_week.style
        .background_gradient(cmap="RdYlGn")
        .format("{:.3f}"),
        use_container_width=True
    )
