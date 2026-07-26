# 🚔 KSP Crime Intelligence Command Centre (KSP-CICC)

An AI-powered crime analytics and visualization platform developed for the **State Crime Records Bureau (SCRB), Karnataka Police Department**. The platform transforms raw FIR records into actionable intelligence using machine learning, geospatial analytics, and interactive dashboards.

---

## 📌 Problem Statement

Traditional police records provide historical information but limited decision-making support. KSP-CICC helps law enforcement identify crime patterns, predict future incidents, detect anomalies, and discover repeat offender behavior using AI-driven analytics.

---

# ✨ Key Features

- 📊 Crime Analytics Dashboard
- 🚨 Anomaly Detection using Hybrid Autoencoder + XGBoost
- 📍 Crime Hotspot Detection using DBSCAN
- 📈 Crime Forecasting using Time Series Models
- 👤 Modus Operandi (MO) Detection for Repeat Offenders
- 🗺 Interactive Maps and Visualizations
- ☁️ Cloud Deployment using Zoho Catalyst

---

# 🏗 System Architecture

```
Police FIR Database
        │
        ▼
Data Cleaning & Feature Engineering
        │
        ▼
Machine Learning Modules
 ├── Anomaly Detection
 ├── Hotspot Detection
 ├── Crime Prediction
 └── MO Detection
        │
        ▼
Processed Datasets
        │
        ▼
Zoho Catalyst Backend
        │
        ▼
React Dashboard
```

---

# 📂 Project Structure

```
AI-Driven_Crime_Analytics_And_Visualization/

│
├── dataset/
│   ├── synthetic_data/
│   ├── training_data/
│   └── processed_data/
│
├── model/
│   ├── anomaly_features.ipynb
│   ├── anomaly_features_improved.ipynb
│   ├── hotspot_detection.ipynb
│   ├── time_series_prediction.ipynb
│   └── mo_detection.ipynb
│
├── frontend/
│
├── backend/
│
├── requirements.txt
│
└── README.md
```

---

# 🧠 Machine Learning Modules

## 1️⃣ Anomaly Detection

### Objective

Identify unusual crime activity and abnormal station-level patterns.

### Model

- Autoencoder
- XGBoost
- Hybrid Autoencoder + XGBoost

### Features Used

- Total Cases
- Heinous Crime Ratio
- Undetected Ratio
- False Case Ratio
- Average Reporting Delay
- Night Incident Percentage
- Crime Diversity
- Arrest Rate

### Output

```
PoliceStationID
Year-Month
Anomaly Score
Risk Probability
```

---

## 2️⃣ Crime Hotspot Detection

### Objective

Identify crime-prone geographical locations.

### Algorithm

DBSCAN

### Features

- Latitude
- Longitude
- Crime Type
- Month
- Time Bucket

### Output

- Cluster ID
- Hotspot Coordinates
- Number of Crimes

---

## 3️⃣ Crime Prediction

### Objective

Forecast future crime volume.

### Models

- Prophet
- ARIMA
- LightGBM
- XGBoost

### Output

```
Station
Crime Type
Month
Predicted Crime Count
```

---

## 4️⃣ Modus Operandi Detection

### Objective

Cluster repeat offenders based on behavioral patterns.

### Algorithm

Feature-space DBSCAN

### Features

- Time of Crime
- Weekend Ratio
- Victim Profile
- Geographic Spread
- Crime Sections
- Arrest/Surrender Ratio

### Output

```
Person UID
Cluster ID
Behavior Profile
```

---

# 🛠 Technology Stack

| Category | Technologies |
|-----------|--------------|
| Frontend | React.js, Tailwind CSS |
| Backend | Python, Node.js |
| Cloud | Zoho Catalyst |
| Database | Catalyst Data Store |
| Storage | Catalyst Stratus |
| ML | TensorFlow, PyTorch, XGBoost, Scikit-Learn |
| Data Processing | Pandas, NumPy |
| GIS | DBSCAN, Shapely |

---

# 📊 Data Pipeline

```
Raw FIR Data
      │
      ▼
Data Cleaning
      │
      ▼
Feature Engineering
      │
      ▼
Machine Learning
      │
      ▼
Predictions
      │
      ▼
Dashboard
```

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/<your-username>/AI-Driven_Crime_Analytics_And_Visualization.git
```

Move into the project

```bash
cd AI-Driven_Crime_Analytics_And_Visualization
```

Create a virtual environment

```bash
python -m venv venv
```

Activate environment

Windows

```bash
venv\Scripts\activate
```

Linux/Mac

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶ Running the Project

Run Jupyter notebooks

```bash
jupyter notebook
```

or

Run backend

```bash
python app.py
```

Run React frontend

```bash
npm install
npm start
```

---

# 📈 Model Summary

| Module | Algorithm | Output |
|----------|-----------|--------|
| Anomaly Detection | Hybrid Autoencoder + XGBoost | Station Risk Score |
| Hotspot Detection | DBSCAN | Crime Clusters |
| Crime Prediction | Prophet / ARIMA / LightGBM | Future Crime Forecast |
| MO Detection | DBSCAN | Offender Behavior Clusters |

---

# ☁ Deployment

- Zoho Catalyst AppSail
- Catalyst Functions
- Catalyst Data Store
- Catalyst Stratus

---

# 📌 Future Enhancements

- Real-time Crime Monitoring
- NLP-based FIR Parsing
- Patrol Route Optimization
- Mobile Officer Application
- LLM-powered Crime Report Summarization

---

# 👥 Contributors

Developed as part of the **AI-Driven Crime Analytics & Visualization** project for the Karnataka State Police (SCRB).

---

# 📄 License

This project is intended for educational and research purposes.
