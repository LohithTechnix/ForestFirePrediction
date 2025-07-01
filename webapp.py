from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

application = Flask(__name__)
app = application

# Load the Ridge model and scaler
ridge_model = pickle.load(open('C:\\Users\\kolli\\OneDrive\\e2eml_project\\pickle\\ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('C:\\Users\\kolli\\OneDrive\\e2eml_project\\pickle\\scaler.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get form data
            data = request.form.to_dict()
            data = {k: float(v) for k, v in data.items()}

            # Create DataFrame
            input_data = pd.DataFrame([data])

            # Scale the input data
            scaled_data = standard_scaler.transform(input_data)

            # Make prediction
            prediction = ridge_model.predict(scaled_data)

            # Prepare response
            response = {
                'prediction': prediction[0]
            }
            return jsonify(response)
        except Exception as e:
            return jsonify({'error': str(e)})

    return render_template('index1.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)

