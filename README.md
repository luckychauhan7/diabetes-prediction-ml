# Diabetes Risk Prediction with Random Forest  

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)  
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.2-orange?logo=scikitlearn)  
![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-brightgreen?logo=streamlit)  
![pandas](https://img.shields.io/badge/pandas-2.2.3-purple?logo=pandas)  
![License](https://img.shields.io/badge/License-MIT-green)  

A complete machine learning pipeline for predicting diabetes risk using the **Pima Indians Diabetes dataset**.  

This project features:  
- A Random Forest classifier for baseline modeling  
- Standardized preprocessing with `StandardScaler`  
- Reproducible training and evaluation  
- A **CLI tool** for quick predictions  
- A **Streamlit dashboard** for interactive exploration and real-time inference  

⚠️ **Note:** This project is intended for **learning and benchmarking** only. It is **not** for clinical or medical use.  

---

## 📌 Overview  

**Goal**  
Predict the onset of diabetes from 8 health features and return an interpretable risk score.  

**Approach**  
- Train/test split  
- Feature standardization  
- Random Forest classification  
- Model persistence with Joblib  
- Simple deployment surfaces (CLI + Streamlit)  

**Outcome**  
Baseline accuracy in the **75–80% range** on held-out test data.  
Predictions include probability outputs to communicate risk, which aligns with common educational baselines reported in tutorials.  

---

## ✨ Key Features  

- 🌲 **Random Forest** with 100 trees and probability outputs (`predict_proba`) for risk communication  
- 🧰 End-to-end scripts: training, saving artifacts, and single-sample prediction via CLI  
- 🖥️ **Streamlit app**: dataset overview, feature distributions, correlations, and on-page predictions in a simple UI  

---

## 📊 Dataset  

- **Source:** Pima Indians Diabetes Database (768 rows, 8 predictors + Outcome)  
- **Features:**  
  - Pregnancies  
  - Glucose  
  - BloodPressure  
  - SkinThickness  
  - Insulin  
  - BMI  
  - DiabetesPedigreeFunction  
  - Age  
- **Target:** `Outcome` ∈ {0,1}  

---

## 📂 Project Structure  

- **Training:**  
  - Load dataset  
  - Scale features with `StandardScaler`  
  - Split into train/test sets  
  - Train Random Forest model  
  - Report accuracy, confusion matrix, and classification report  
  - Save model & scaler with Joblib  

- **Prediction (CLI):**  
  - Load saved model & scaler  
  - Accept typed input values  
  - Apply scaling  
  - Return class label + probability  

- **Streamlit app:**  
  - Show dataset stats, histograms, and correlation heatmap  
  - Provide instant predictions from user input  

---

## 🚀 How to Run  

1. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
