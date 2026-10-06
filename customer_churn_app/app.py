from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("churn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get 6 inputs from form
    tenure = float(request.form['tenure'])
    MonthlyCharges = float(request.form['MonthlyCharges'])
    TotalCharges = float(request.form['TotalCharges'])
    SeniorCitizen = float(request.form['SeniorCitizen'])
    Contract = float(request.form['Contract'])
    InternetService = float(request.form['InternetService'])

    # Convert to array
    features = np.array([[tenure, MonthlyCharges, TotalCharges, SeniorCitizen, Contract, InternetService]])

    # Scale
    features_scaled = scaler.transform(features)

    # Predict
    prediction = model.predict(features_scaled)[0]
    output = "Customer WILL STAY 😊" if prediction == 0 else "Customer WILL CHURN 😟"

    return render_template('index.html', prediction_text=f"Prediction: {output}")

if __name__ == "__main__":
    app.run(debug=True)
