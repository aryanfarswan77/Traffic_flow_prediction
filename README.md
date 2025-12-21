# 🚦 Traffic Flow Prediction Using LSTM

## 📌 Overview
Traffic congestion is a major challenge in urban areas. This project implements a **deep learning–based traffic flow prediction system** using an **LSTM (Long Short-Term Memory) neural network** to forecast **next-hour traffic queue density** based on historical data.

The model captures **temporal patterns and periodicity** in traffic flow using **cyclical time feature encoding** and is deployed through an **interactive Streamlit web application** for real-time prediction and visualization.


---

## 🎯 Objectives
- Predict short-term traffic congestion accurately  
- Capture daily and weekly traffic patterns using time-series modeling  
- Provide an interactive and user-friendly web interface  
- Ensure consistency between model training and deployment  

---

## 📊 Dataset

- **Dataset Name:** Delhi Traffic Density Dataset  
- **Source:**  
  https://delhi-trafficdensity-dataset.github.io/

### Dataset Description
The dataset contains traffic density information collected from multiple regions of Delhi. It includes timestamped traffic data which is suitable for **time-series forecasting and congestion analysis**.

The raw dataset is preprocessed to:
- Resample data to hourly intervals  
- Handle missing values  
- Generate derived features for model training  

---

## 🧠 Methodology

### 1️⃣ Data Preprocessing
- Raw traffic data is cleaned and resampled to **hourly intervals**
- Missing values are handled appropriately
- Feature engineering is performed:
  - `avg_queue_density` (target)
  - `is_weekend`
  - `is_peak_hour`
  - **Cyclical encoding of time**:
    - `hour_sin`, `hour_cos`
    - `dow_sin`, `dow_cos`
- Data is scaled using **MinMaxScaler**
- Fixed-length sequences of **48 hours** are created for LSTM input

---

### 2️⃣ Model Training
- An **LSTM-based regression model** is trained on historical traffic sequences
- Loss function: **Mean Squared Error (MSE)**
- Validation split is used to monitor overfitting
- Trained model is saved in **`.keras` format**
- Scaler and metadata are stored to ensure reproducibility

---

### 3️⃣ Web Application
- Built using **Streamlit**
- Users can:
  - Select date and hour
  - View historical traffic trends
  - Predict next-hour traffic density
  - Receive congestion alerts (Low / Moderate / High)
  - Visualize traffic heatmaps (daily & weekly)

---

## 🏗 System Architecture

### 🔹 Overall Architecture
    ┌──────────────────────┐
    │   Traffic Dataset     │
    │ (Delhi Traffic Data)  │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │  Data Preprocessing   │
    │ - Cleaning            │
    │ - Resampling (Hourly) │
    │ - Feature Engineering │
    │ - Cyclical Encoding   │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │  Feature Scaling      │
    │  (MinMaxScaler)       │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │  Sequence Generation  │
    │  (48-hour windows)    │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │   LSTM Model Training │
    │   (TensorFlow/Keras)  │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │  Saved Model (.keras) │
    │  + Scaler + Metadata  │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │  Streamlit Web App    │
    │  - User Input         │
    │  - Prediction         │
    │  - Visualization     │
    └──────────────────────┘

---

## 🗂 Project Structure
traffic-flow-prediction/
│
├── data/
│ └── hourly_processed.csv
│
├── models/
│ └── lstm_traffic_model.keras
│
├── preprocessing.ipynb
├── train_model.ipynb
├── app.py
├── scaler.pkl
├── metadata.pkl
├── X_train.npy
├── y_train.npy
├── X_test.npy
├── y_test.npy
└── README.md


---

## 🛠 Tech Stack
- **Python**
- **TensorFlow / Keras**
- **NumPy, Pandas**
- **Scikit-learn**
- **Streamlit**

---

## 📊 Output & Features

- Normalized traffic density prediction (0–1)
- Traffic level classification:
  - 🟢 Low Traffic
  - 🟡 Moderate Traffic
  - 🔴 High Traffic
- Hour-wise and weekly traffic heatmaps
- Peak-hour congestion alerts

---

## 🎓 Key Learnings

- Time-series forecasting using LSTM
- Importance of cyclical feature encoding
- Maintaining feature consistency between training and inference
- End-to-end ML pipeline deployment

---

## 🔮 Future Enhancements

- Real-time traffic data integration via APIs
- Multi-location traffic prediction
- Advanced architectures (GRU, Transformers)
- Cloud deployment (AWS / GCP / Azure)
- Mobile-friendly UI

---

## 👨‍💻 Author

**Aryan Farswan**  
B.Tech – Artificial Intelligence & Machine Learning  
contact- [LinkedIn](https://www.linkedin.com/in/aryan-farswan-29a09431a)

