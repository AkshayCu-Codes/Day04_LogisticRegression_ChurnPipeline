# 📌 Day 04 — Logistic Regression Customer Churn Prediction Pipeline

This project extends **Day 03 (KNN)** by training a **Logistic Regression** model on the same Telco Churn dataset to allow multi-model comparison for Day 05. The result is a deployable ML pipeline that can be integrated with an API and dashboard.

---

## 🎯 Project Objectives
- Train a Logistic Regression churn model
- Apply full preprocessing: Scaling + OneHotEncoding
- Perform hyperparameter tuning with GridSearchCV
- Evaluate metrics: Accuracy, Precision/Recall, ROC-AUC
- Export model for API + UI deployment
- Prepare for multi-model leaderboard (Day 05)

---

## 📂 Folder Structure
```
Day04_LogReg_CustomerChurnPipeline/
│
├── notebooks/
│   └── Day04_LogReg.ipynb
├── model/
│   └── churn_logreg_model.pkl 
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
└── README.md
```

---

## 🧠 Dataset Used
**Telco Customer Churn Dataset**  
Source: Kaggle  
Same dataset used in Day 03 to ensure model comparison is fair and consistent.

---

## ⚙️ Setup & Installation

### Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate       # Windows
```

### Install dependencies
```bash
pip install scikit-learn pandas numpy seaborn matplotlib joblib
```

---

## 🚀 Run the Notebook
Open:
```
notebooks/Day04_LogReg.ipynb
```

Run all cells to:
- Load data
- Preprocess features
- Train & tune model
- Evaluate metrics
- Export `.pkl` file

---

## 📊 Model Evaluation Metrics

| Metric            | Purpose |
|-------------------|----------|
| Accuracy          | Baseline performance |
| Precision/Recall  | Retention decision impact |
| Confusion Matrix  | Misclassification insight |
| ROC-AUC Score     | Ranking churn probability strength |

output:
```
Accuracy: ~0.79 
AUC Score: ~0.83
```

---

## 💾 Model Export
The trained model is saved as:
```
model/churn_logreg_model.pkl
```

This file will be used for:
- FastAPI `/predict` endpoint
- Model selection in Streamlit dashboard
- Day 05 model leaderboard

Ensure folder exists before saving:
```python
import os
os.makedirs("model", exist_ok=True)
```
