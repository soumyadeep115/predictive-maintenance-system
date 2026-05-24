# Predictive Maintenance System using Machine Learning

## Overview
This project is a machine learning-based predictive maintenance system designed to analyze machine sensor data and predict potential machine failure conditions.

The system uses a **Random Forest Classifier with SMOTE balancing** to detect abnormal operating conditions and classify machine health into:

- **Normal Operation**
- **Warning: Potential Failure**
- **Critical Failure Risk**

A real-time interactive dashboard was developed using **Streamlit**, and a prediction API was deployed using **FastAPI**.

---

## Features

- Machine sensor data analysis
- Failure prediction using ML
- Class imbalance handling using SMOTE
- Interactive Streamlit dashboard
- FastAPI REST API deployment
- Real-time sensor input prediction
- Visual sensor trend monitoring
- Multi-level risk detection

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Streamlit
- FastAPI
- Uvicorn
- Matplotlib
- Seaborn
- Pickle

---

## Dataset

The project uses machine sensor data containing:

- Air Temperature [K]
- Process Temperature [K]
- Rotational Speed [rpm]
- Torque [Nm]
- Tool Wear [min]
- Machine Type
- Failure Target

---

## System Architecture

The project follows a three-layer architecture:

**Data Layer**
- Machine sensor dataset (CSV)

**Machine Learning Layer**
- Data preprocessing
- SMOTE balancing
- Random Forest model training
- Model serialization using Pickle

**Application Layer**
- Streamlit dashboard for real-time predictions
- FastAPI REST API for external access
- Prediction engine using trained model

## Machine Learning Workflow

### 1. Data Preprocessing
- Removed unnecessary columns
- Encoded categorical machine type
- Split dataset into training and testing sets

### 2. Exploratory Data Analysis
- Correlation heatmap
- Sensor trend analysis
- Failure distribution visualization

### 3. Model Training
A **Random Forest Classifier** was used.

To address class imbalance, **SMOTE** was applied.

### Why SMOTE?
SMOTE was selected because the dataset had significantly fewer failure cases compared to normal cases. It improved failure recall from the baseline model, allowing better detection of abnormal machine behavior.

---

## Model Performance

### Baseline Model
Accuracy: 98.1%

### Final SMOTE Model
Accuracy: 95.6%

The final model was selected because it improved failure detection capability, which is more important for predictive maintenance systems.

---


## Challenges Faced

During development, several challenges were encountered:

- Handling severe class imbalance in failure prediction
- Improving failure recall without sacrificing overall accuracy
- Integrating the trained model with FastAPI and Streamlit
- Designing realistic threshold-based risk classification
- Ensuring consistent predictions across dashboard and API

---

---

## Future Improvements

Future enhancements for this project include:

- Live IoT sensor integration
- Remaining Useful Life (RUL) estimation
- Cloud deployment for remote monitoring
- Historical data logging and analytics
- Automated maintenance alert notifications

--- 

## Running the Project

### Start the Streamlit Dashboard

Run:

```bash
streamlit run app.py
```

This launches the interactive predictive maintenance dashboard where sensor values can be entered to view live failure predictions.

---

### Start the FastAPI Server

Open another terminal and run:

```bash
uvicorn api:app --reload
```

Once the server starts, manually open the following URL in your browser:

```bash
http://127.0.0.1:8000/docs
```

This opens the FastAPI Swagger documentation interface where prediction requests can be tested interactively.



