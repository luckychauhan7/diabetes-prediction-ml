# diabetes-prediction-ml
Machine learning project for diabetes risk prediction using Random Forest

--A complete ML pipeline for diabetes risk prediction on the Pima Indians Diabetes dataset featuring a Random Forest classifier, standardized preprocessing, reproducible training/evaluation, a CLI predictor for one‑off inputs, and a Streamlit dashboard for data exploration and real‑time inference; intended for learning and baseline benchmarking, not for clinical use. --



Overview

Goal: Predict diabetes onset from 8 features in the Pima Indians Diabetes dataset and surface an interpretable risk score for decision support 🩺📊.

Approach: Train/test split with standardized features, Random Forest classification, model persistence (Joblib), and simple deployment surfaces (CLI + Streamlit) for fast experimentation and use ⚙️🧪.

Outcome: Baseline accuracy in the 75–80% range on held‑out data, with probability outputs to communicate risk—consistent with common educational baselines reported across community tutorials ✅.

Key features

🌲 Random Forest with 100 trees and calibrated‑style probability outputs via predict_proba for friendly risk communication.

🧰 End‑to‑end scripts: training, artifact saving, and a CLI for single‑sample predictions—great for quick checks and automation.

🖥️ Streamlit app: dataset overview, feature distributions, correlations, and on‑page predictions in a clean UI for non‑technical stakeholders.

Dataset

Source: Pima Indians Diabetes Database (768 rows; 8 predictors + Outcome), a widely used benchmark; patients are adult females of Pima Indian heritage 📚.

Features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age; target Outcome ∈ {0,1} 🧾.

Project structure

Training: Read CSV, scale features with StandardScaler, split train/test, fit Random Forest, and report accuracy, confusion matrix, and classification report—then save model/scaler with Joblib 🧪.

Prediction (CLI): Load artifacts, accept typed inputs, apply the same scaling, and return class label plus probability 🔍.

Streamlit app: Load data and artifacts, show dataset stats, histograms, and a correlation heatmap, then generate instant predictions from inputs 🎛️.

How to run

Install deps and ensure diabetes.csv is available—or pull the canonical CSV URL used across community examples 📥.

Train: python train_model.py to produce diabetes_prediction_model.pkl and scaler.pkl in the project root 🧱.

Predict (CLI): python predict_cli.py to enter values interactively and view label + probability in the terminal ▶️.

Launch app: streamlit run Diabetes-Prediction-by-Lucky-Chauhan.py to explore data and make on‑page predictions 🌐.

Modeling notes

Why Random Forest: strong tabular baseline, resilient to non‑linearities and feature interactions, and easy to interpret via feature importance 🌲.

Typical baselines: community work on this dataset often lands mid‑70s to mid‑80s without heavy tuning; gains come from better imputation, class weighting, and hyperparameter search 🎯.

Limitations

Educational use only: small dataset with known quirks (e.g., zeros in physiological fields), so predictions are not clinical advice 🧪🚫.

Cohort constraints: trained on adult Pima Indian females; broader use requires retraining and validation on representative populations 🌍.

Roadmap

🔎 Add cross‑validation and RandomizedSearchCV to tighten performance bounds and reduce variance.

🎯 Calibrate probabilities (Platt/Isotonic) to improve risk communication, especially in the UI.

🪄 Add explainability (feature importances/SHAP) inside the Streamlit app for transparency and trust.

References

Pima Indians Diabetes Database overview and canonical CSV access 🔗.

Comparable Random Forest and Streamlit implementations for diabetes prediction and demo apps 📎.
