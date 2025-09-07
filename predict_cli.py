import pandas as pd
import joblib


model = joblib.load('diabetes_prediction_model.pkl')
scaler = joblib.load('scaler.pkl')

def get_input(prompt, dtype=float):
    while True:
        try:
            value = dtype(input(prompt))
            return value
        except ValueError:
            print("Invalid input. Please enter a valid number.")

def main():
    print("Enter patient details for diabetes prediction:")
    pregnancies = get_input("Pregnancies (e.g., 2): ", int)
    glucose = get_input("Glucose Level (e.g., 120): ")
    blood_pressure = get_input("Blood Pressure (e.g., 70): ")
    skin_thickness = get_input("Skin Thickness (e.g., 20): ")
    insulin = get_input("Insulin (e.g., 80): ")
    bmi = get_input("BMI (e.g., 25.0): ")
    dpf = get_input("Diabetes Pedigree Function (e.g., 0.5): ")
    age = get_input("Age (e.g., 30): ", int)

   
    input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness,
                                insulin, bmi, dpf, age]],
                              columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                                       'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

    input_scaled = scaler.transform(input_data)

    #pedict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    result = "Diabetic" if prediction == 1 else "Non-Diabetic"
    print(f"\nPrediction: {result} (Probability of Diabetes: {probability:.2%})")
    

if __name__ == "__main__":
    main()
