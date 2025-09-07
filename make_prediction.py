import pandas as pd
import joblib

model = joblib.load('diabetes_prediction_model.pkl')
scaler = joblib.load('scaler.pkl')


patient_data = pd.DataFrame([[2, 120, 80, 25, 100, 28.5, 0.3, 35]],
                            columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                                     'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])


scaled_data = scaler.transform(patient_data)
prediction = model.predict(scaled_data)
probability = model.predict_proba(scaled_data)[0][1]  # Probability of diabetes

print(f"Prediction: {'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'}")
print(f"Probability of diabetes: {probability:.2%}")
