import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import warnings

warnings.filterwarnings('ignore', message='X does not have valid feature names')

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
data = pd.read_csv(url, header=None, names=columns)

print("Missing values in the dataset:\n", data.isnull().sum())

X = data.drop('Outcome', axis=1)
y = data['Outcome']

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#wevaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('Confusion Matrix:\n', conf_matrix)
print('Classification Report:\n', class_report)

#save the model and scaler
joblib.dump(model, 'diabetes_prediction_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model and scaler saved as 'diabetes_prediction_model.pkl' and 'scaler.pkl'")

#exxample prediction with proper column names
new_data = pd.DataFrame([[6, 148, 72, 35, 0, 33.6, 0.627, 50]], 
                       columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                               'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)
print("Prediction (1 = Diabetic, 0 = Non-Diabetic):", prediction[0])
