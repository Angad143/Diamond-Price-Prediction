from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('best_model/rf_model.pkl')

# Create route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Create route to handle form submission and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    pass

if __name__ == '__main__':
    app.run(debug=True)
