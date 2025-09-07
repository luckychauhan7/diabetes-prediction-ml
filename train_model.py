import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import warnings

warnings.filterwarnings('ignore', message='X does not have valid feature names')

data = pd.read_csv('diabetes.csv')

print("Dataset loaded successfully!")
print("Missing values in the dataset:")
print(data.isnull().sum())


X = data.drop('Outcome', axis=1)
y = data['Outcome']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#random forest model ttrainning
print("Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2f}')
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('Classification Report:')
print(classification_report(y_test, y_pred))

#saving model and scaler
joblib.dump(model, 'diabetes_prediction_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model and scaler saved as 'diabetes_prediction_model.pkl' and 'scaler.pkl'")
