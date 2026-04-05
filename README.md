# 🩸 AnemiaCare AI: Intelligent Anemia Detection System

## 📌 Overview

This project presents a machine learning-based system for detecting anemia using Complete Blood Count (CBC) data. Unlike traditional diagnosis methods that rely mainly on Hemoglobin (Hb), this system utilizes multiple hematological parameters to improve prediction accuracy and reliability.

The system integrates data preprocessing, model training, evaluation, and deployment into an interactive Streamlit web application that provides real-time predictions, risk analysis, and downloadable reports.

---

## 🎯 Objectives

* Develop a predictive model for anemia diagnosis using supervised learning
* Analyze multiple CBC parameters beyond Hemoglobin
* Provide probability-based risk scoring and confidence levels
* Compare multiple machine learning models
* Build an interactive and user-friendly screening system

---

## ⚠️ Problem Statement

Traditional anemia diagnosis often depends heavily on Hemoglobin levels, which may not capture complex relationships between various blood parameters.

This project aims to:

* Use multiple CBC features
* Provide early and reliable screening
* Enhance decision-making using machine learning

---

## 🔬 Dataset Description

* Records: ~15,000+ samples
* Features: 12 attributes

### Key Attributes:

* Age
* Gender
* Hemoglobin (Hb)
* RBC (Red Blood Cells)
* PCV (Packed Cell Volume)
* MCV, MCH, MCHC
* RDW (Red Cell Distribution Width)
* WBC / TLC
* Platelet Count
* Target (Anemic / Non-Anemic)

---

## ⚙️ System Workflow

User Input (CBC Values)
→ Data Preprocessing
→ ML Model Prediction
→ Anemia Classification
→ Risk Score & Confidence
→ Visualization Dashboard
→ PDF Report Generation

---

## 🧰 Technology Stack

### 🖥️ Core

* Python
* Streamlit

### 📊 Data Processing

* Pandas
* NumPy

### 🤖 Machine Learning

* Scikit-learn
* XGBoost

### 📈 Visualization

* Plotly

### 📄 Reporting

* FPDF

### 💾 Storage

* Pickle

---

## 🤖 Models Implemented

* Random Forest
* Support Vector Machine (SVM)
* Naive Bayes
* AdaBoost
* XGBoost

### Evaluation Scenarios:

1. With Hemoglobin (High accuracy)
2. Without Hemoglobin (Real-world robustness)

---

## 🏆 Final Model Selection

AdaBoost was selected as the final deployed model due to:

* Stable and consistent performance
* Good probability calibration
* Lower computational cost
* Suitability for real-time prediction

---

## 📊 Performance Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

---

## 🔥 Key Features

### ✅ Prediction System

* Input CBC values
* Real-time anemia detection

### ✅ Risk Analysis

* Probability-based risk score
* Model confidence

### ✅ Visualization

* Gauge chart (risk %)
* Feature comparison graphs

### ✅ Model Comparison

* Multi-model evaluation
* With/without Hb analysis

### ✅ Dataset Explorer

* Data preview & statistics
* Missing value detection
* Feature distribution

### ✅ Analytics Dashboard

* ROC Curve
* Confusion Matrix

### ✅ PDF Report Generation

* Patient details
* Prediction results
* Risk score & recommendations

---

## 🧪 Prediction Logic

```python
proba = model.predict_proba(input_data)[0]
confidence = max(proba) * 100
risk_score = proba[1] * 100
```

---

## 📂 Project Structure

```
anemia-detection-ml/
│
├── anemia.py
├── data_cleaning.py
├── final_model_comparison.py
├── train_all_models_with_hgb.py
├── train_all_models_without_hgb.py
│
├── models/
├── dataset/
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
streamlit run anemia.py
```

---

## 🧠 Advantages

* Early anemia detection
* Multi-parameter analysis
* Fast and automated predictions
* User-friendly interface

---

## ⚠️ Limitations

* Depends on dataset quality
* Not a replacement for clinical diagnosis
* Requires accurate inputs

---

## 🚀 Future Scope

* Healthcare system integration
* Mobile application
* Deep learning models
* Personalized treatment suggestions

---

## 🏁 Conclusion

This project demonstrates how supervised machine learning can enhance anemia detection using hematological data. It provides accurate predictions, risk analysis, and visual insights, making it a useful decision-support tool.

---

## 💬 Final Statement

"This project integrates machine learning with hematological analysis to provide an intelligent, real-time anemia screening system."

Ajay
B.Tech IT (Final Year)
