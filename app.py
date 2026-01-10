import streamlit as st
import numpy as np
import pandas as pd
import pickle
import base64
from pathlib import Path
from tensorflow.keras.models import load_model
from datetime import datetime, time, timedelta

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Delhi Traffic Flow Prediction",
    page_icon="🚦",
    layout="centered"
)

# --------------------------------------------------
# Load Background Image (Base64 - LOCAL FILE)
# --------------------------------------------------
def get_base64_bg_image(image_path):
    img_bytes = Path(image_path).read_bytes()
    return base64.b64encode(img_bytes).decode()

bg_image = get_base64_bg_image("assets/city_bg.webp")

# --------------------------------------------------
# Apply Background CSS
# --------------------------------------------------
st.markdown(f"""
<style>

/* ===== MAIN BACKGROUND WITH CITY IMAGE ===== */
.stApp {{
    background:
        linear-gradient(
            rgba(2, 6, 23, 0.85),
            rgba(2, 6, 23, 0.85)
        ),
        url("data:image/webp;base64,{bg_image}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    color: #E6EDF3;
}}

/* ===== HEADINGS ===== */
h1, h2, h3 {{
    color: #F8FAFC;
    font-weight: 700;
}}

/* ===== SUBHEADERS ===== */
.css-10trblm {{
    color: #38BDF8 !important;
}}

/* ===== INPUT FIELDS ===== */
input, textarea, select {{
    background-color: rgba(15, 23, 42, 0.92) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(148,163,184,0.35) !important;
    border-radius: 10px !important;
}}

/* ===== BUTTON ===== */
.stButton>button {{
    background: linear-gradient(135deg,
        #22C55E,
        #FACC15,
        #EF4444
    );
    color: #020617;
    font-weight: 700;
    border-radius: 12px;
    padding: 0.7em 1.8em;
    border: none;
    transition: 0.3s ease;
}}

.stButton>button:hover {{
    transform: scale(1.06);
    box-shadow: 0px 0px 20px rgba(56,189,248,0.6);
}}

/* ===== METRIC CARDS ===== */
[data-testid="metric-container"] {{
    background: rgba(2, 6, 23, 0.88);
    border-radius: 14px;
    padding: 18px;
    border: 1px solid rgba(148,163,184,0.25);
    box-shadow: 0 10px 30px rgba(0,0,0,0.7);
}}

/* ===== STATUS BOXES ===== */
.stSuccess {{
    background-color: rgba(34,197,94,0.15);
    border-left: 5px solid #22C55E;
}}

.stWarning {{
    background-color: rgba(250,204,21,0.15);
    border-left: 5px solid #FACC15;
}}

.stError {{
    background-color: rgba(239,68,68,0.15);
    border-left: 5px solid #EF4444;
}}

.stInfo {{
    background-color: rgba(56,189,248,0.15);
    border-left: 5px solid #38BDF8;
}}

/* ===== CHART CONTAINERS ===== */
[data-testid="stLineChart"] {{
    background-color: rgba(2, 6, 23, 0.88);
    border-radius: 14px;
    padding: 12px;
}}

/* ===== TABLE / HEATMAP ===== */
.css-1lcbmhc, .css-1n76uvr {{
    background-color: rgba(2, 6, 23, 0.88) !important;
    color: #E6EDF3 !important;
}}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load Model & Scaler
# --------------------------------------------------
model = load_model("models/lstm_traffic_model.keras", compile=False)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

SEQ_LEN = 48

# --------------------------------------------------
# Load Data
# --------------------------------------------------
hourly_df = pd.read_csv(
    "data/ui_ready_traffic_data.csv",
    index_col=0,
    parse_dates=True
).reset_index().rename(columns={"index": "timestamp"})


# --------------------------------------------------
# Helper Function
# --------------------------------------------------
def traffic_card(value):
    if value < 0.33:
        return (
            "🟢 Low Traffic",
            "Smooth traffic conditions. Ideal time to travel.",
            "Low congestion expected. Roads are likely to be free-flowing."
        )
    elif value < 0.66:
        return (
            "🟡 Moderate Traffic",
            "Average traffic conditions. Expect minor delays.",
            "Moderate congestion. Leave a little earlier to avoid delays."
        )
    else:
        return (
            "🔴 High Traffic",
            "Heavy traffic expected. Consider alternate routes.",
            "Traffic congestion is usually higher during this time."
        )

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown("""
<h1 style='text-align:center;'>🚦 Delhi Traffic Flow Prediction</h1>
<p style='text-align:center; font-size:18px; color:#CBD5E1;'>
Predict traffic density in advance and plan smarter travel
</p>
<hr>
""", unsafe_allow_html=True)

# --------------------------------------------------
# User Input
# --------------------------------------------------
st.subheader("📅 Select Date & Time for Travel")

col1, col2 = st.columns(2)

with col1:
    selected_date = st.date_input(
        "Select Date",
        value=hourly_df["timestamp"].dt.date.max()
    )

with col2:
    selected_hour = st.selectbox(
        "Select Hour (24-hour format)",
        [f"{h:02d}:00" for h in range(24)],
        index=15
    )

predict_btn = st.button("🚀 Predict Traffic")

# --------------------------------------------------
# Prediction Logic (UNCHANGED)
# --------------------------------------------------
if predict_btn:

    hour_int = int(selected_hour.split(":")[0])
    selected_datetime = datetime.combine(selected_date, time(hour_int, 0))
    last_known_time = hourly_df["timestamp"].max()
    is_future_date = selected_datetime.date() > last_known_time.date()

    if selected_datetime <= last_known_time:
        history = hourly_df[hourly_df["timestamp"] < selected_datetime].tail(SEQ_LEN)
        steps = 1
    else:
        history = hourly_df.tail(SEQ_LEN)
        steps = int((selected_datetime - last_known_time).total_seconds() // 3600)
        steps = max(1, min(steps, 24))

    recent_values = history["avg_queue_density"].values
    scaled_input = scaler.transform(recent_values.reshape(-1, 1))
    input_seq = scaled_input.reshape(1, SEQ_LEN, 1)

    preds = []
    for _ in range(steps):
        p = model.predict(input_seq, verbose=0)[0][0]
        preds.append(p)
        input_seq = np.append(input_seq[:, 1:, :], [[[p]]], axis=1)

    final_prediction = scaler.inverse_transform(
        np.array([[preds[-1]]])
    )[0][0]
    predicted_day_df = None

    if is_future_date:
        # Start from last known sequence
        recent_seq = hourly_df.tail(SEQ_LEN)["avg_queue_density"].values
        recent_scaled = scaler.transform(recent_seq.reshape(-1, 1))

        future_hours = 24
        preds = []
        current_seq = recent_scaled.copy()

        for _ in range(future_hours):
            p = model.predict(
                current_seq.reshape(1, SEQ_LEN, 1),
                verbose=0
            )[0][0]
            preds.append(p)
            current_seq = np.append(current_seq[1:], [[p]], axis=0)

        preds = scaler.inverse_transform(
            np.array(preds).reshape(-1, 1)
        ).ravel()

        future_times = [
            datetime.combine(selected_date, time(0, 0)) + timedelta(hours=i)
            for i in range(future_hours)
        ]

        predicted_day_df = pd.DataFrame({
            "timestamp": future_times,
            "avg_queue_density": preds
        })


    level, short_msg, advice_msg = traffic_card(final_prediction)

    st.markdown("## 🚦 Traffic Prediction Result")

    colA, colB = st.columns(2)

    with colA:
        st.metric("Predicted Avg Queue Density", f"{final_prediction:.3f}")

    with colB:
        if "Low" in level:
            st.success(level)
        elif "Moderate" in level:
            st.warning(level)
        else:
            st.error(level)

    st.info(short_msg)

    st.markdown(f"""
    <div style="font-size:18px;font-weight:600;padding:14px;
    border-radius:10px;background-color:rgba(255,255,255,0.05);">
    {advice_msg}
    </div>
    """, unsafe_allow_html=True)

    st.caption(
        f"🕒 Prediction for {selected_datetime.strftime('%d %b %Y, %H:%M')}"
    )

    # --------------------------------------------------
    # Line Chart
    # --------------------------------------------------
    st.markdown("## 📈 Traffic Trend for Selected Date")

    day_start = datetime.combine(selected_date, time(0, 0))
    day_end = day_start + timedelta(days=1)

    if is_future_date and predicted_day_df is not None:
        day_data = predicted_day_df
    else:
        day_data = hourly_df[
            (hourly_df["timestamp"] >= day_start) &
            (hourly_df["timestamp"] < day_end)
        ]


    if not day_data.empty:
        st.line_chart(day_data.set_index("timestamp")["avg_queue_density"])
    # --------------------------------------------------
    # Intra-Day Traffic Insight (EXPLAINS THE LINE GRAPH)
    # --------------------------------------------------
    day_start_density = day_data.iloc[0]["avg_queue_density"]
    day_end_density = day_data.iloc[-1]["avg_queue_density"]

    if day_end_density > day_start_density:
        st.info("📅 **Intra-day outlook:** Traffic is expected to increase as the day progresses.")
    else:
        st.info("📅 **Intra-day outlook:** Traffic is expected to ease as the day progresses.")


    # --------------------------------------------------
    # Heatmap
    # --------------------------------------------------
    st.markdown("## 🔥 Traffic Density Heatmap (7-Day Context)")

    # Ensure hour column exists
    hourly_df["hour"] = hourly_df["timestamp"].dt.hour

    # 🔑 FIX: If selected date is in the future, anchor heatmap to last known date
    anchor_date = min(
        selected_datetime.date(),
        last_known_time.date()
    )

    heatmap_start = anchor_date - timedelta(days=6)

    heatmap_df = hourly_df[
        (hourly_df["timestamp"].dt.date >= heatmap_start) &
        (hourly_df["timestamp"].dt.date <= anchor_date)
    ].copy()

    heatmap_df["date"] = heatmap_df["timestamp"].dt.date

    # Pivot for heatmap
    heatmap_data = heatmap_df.pivot_table(
        values="avg_queue_density",
        index="date",
        columns="hour"
    )

    # Display heatmap table
    st.dataframe(
        heatmap_data.style
        .background_gradient(cmap="RdYlGn")
        .format("{:.3f}"),
        use_container_width=True
    )

    
    # ==================================================
    # NEW ADDITIONS (Graphs 6, 7, 8 + Future Trend)
    # ==================================================

    # --------------------------------------------------
    # GRAPH 6: Queue vs Stop Density (Selected Date)
    # --------------------------------------------------
    st.markdown("### 🚗 Queue vs Stop Density Contribution")

    if not is_future_date:
        # Historical view
        day_data_adv = hourly_df[
            (hourly_df["timestamp"] >= day_start) &
            (hourly_df["timestamp"] < day_end)
        ]

        if not day_data_adv.empty and "avg_stop_density" in hourly_df.columns:
            st.line_chart(
                day_data_adv
                .set_index("timestamp")[["avg_queue_density", "avg_stop_density"]]
            )

    else:
        # Future view (predicted queue + historical stop context)
        if "avg_stop_density" in hourly_df.columns:

            # Build predicted queue density for full day
            future_hours = 24
            recent_seq = hourly_df.tail(SEQ_LEN)["avg_queue_density"].values
            recent_scaled = scaler.transform(recent_seq.reshape(-1, 1))

            preds = []
            current_seq = recent_scaled.copy()

            for _ in range(future_hours):
                p = model.predict(
                    current_seq.reshape(1, SEQ_LEN, 1),
                    verbose=0
                )[0][0]
                preds.append(p)
                current_seq = np.append(current_seq[1:], [[p]], axis=0)

            predicted_queue = scaler.inverse_transform(
                np.array(preds).reshape(-1, 1)
            ).ravel()

            future_times = [
                datetime.combine(selected_date, time(0, 0)) + timedelta(hours=i)
                for i in range(future_hours)
            ]

            # Historical stop density average by hour
            hourly_stop_avg = (
                hourly_df
                .groupby(hourly_df["timestamp"].dt.hour)["avg_stop_density"]
                .mean()
            )

            predicted_stop = [
                hourly_stop_avg.get(t.hour, hourly_stop_avg.mean())
                for t in future_times
            ]

            future_qs_df = pd.DataFrame({
                "timestamp": future_times,
                "avg_queue_density": predicted_queue,
                "avg_stop_density": predicted_stop
            })

            st.line_chart(
                future_qs_df
                .set_index("timestamp")[["avg_queue_density", "avg_stop_density"]]
            )
        else:
            st.info("ℹ️ Stop density data not available for this dataset.")


    # --------------------------------------------------
    # GRAPH 7: Weekly Congestion Pattern (Selected Hour)
    # --------------------------------------------------
    st.markdown("### 📊 Weekly Congestion Pattern at Selected Time")

    hourly_df["day_of_week"] = hourly_df["timestamp"].dt.day_name()
    hourly_df["hour"] = hourly_df["timestamp"].dt.hour

    weekly_df = hourly_df[
        hourly_df["hour"] == hour_int
    ]

    weekly_avg = (
        weekly_df
        .groupby("day_of_week")["avg_queue_density"]
        .mean()
        .reindex([
            "Monday", "Tuesday", "Wednesday",
            "Thursday", "Friday", "Saturday", "Sunday"
        ])
    )

    st.bar_chart(weekly_avg)

    # --------------------------------------------------
    # GRAPH 8: Unusual Congestion Indicator
    # --------------------------------------------------
    st.markdown("### ⚠️ Unusual Traffic Condition Indicator")

    selected_day_name = selected_datetime.strftime("%A")

    # Get historical average for the SAME weekday at this hour
    baseline_value = weekly_avg.loc[selected_day_name]

    current_density = final_prediction

    # Compare against same-day baseline
    if current_density > baseline_value * 1.20:
        st.warning(
            "⚠️ Traffic congestion is significantly higher than usual "
            f"for a typical {selected_day_name} at this time."
        )
    elif current_density < baseline_value * 0.80:
        st.success(
            "✅ Traffic congestion is lower than usual "
            f"for a typical {selected_day_name} at this time."
        )
    else:
        st.info(
            "ℹ️ Traffic congestion is within the normal range "
            f"for a typical {selected_day_name} at this time."
        )


    # # --------------------------------------------------
    # # FUTURE TRAFFIC FORECAST + TREND MESSAGE
    # # --------------------------------------------------
    # st.markdown("### 🔮 Short-Term Traffic Forecast")

    # # Prepare recent sequence (last 48 points)
    # recent_values = hourly_df["avg_queue_density"].values[-SEQ_LEN:]
    # recent_scaled = scaler.transform(recent_values.reshape(-1, 1))

    # # Multi-step forecast (next 8 hours)
    # future_scaled = []
    # current_seq = recent_scaled.copy()

    # for _ in range(8):
    #     pred = model.predict(
    #         current_seq.reshape(1, SEQ_LEN, 1),
    #         verbose=0
    #     )[0][0]
    #     future_scaled.append(pred)
    #     current_seq = np.append(current_seq[1:], [[pred]], axis=0)

    # future_pred = scaler.inverse_transform(
    #     np.array(future_scaled).reshape(-1, 1)
    # ).ravel()

    # # Increase / Decrease Message
    # if future_pred[0] > current_density:
    #     st.info("📈 Traffic is expected to increase in the coming hours.")
    # else:
    #     st.info("📉 Traffic is expected to decrease in the coming hours.")

    # # Forecast Line Chart
    # forecast_df = pd.DataFrame({
    #     "Hour Ahead": [f"+{i+1}h" for i in range(len(future_pred))],
    #     "Predicted Traffic Density": future_pred
    # }).set_index("Hour Ahead")

    # st.line_chart(forecast_df)
    
