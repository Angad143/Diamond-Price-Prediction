from flask import Flask, render_template, request, redirect, url_for, flash
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

# Define the Flask application and set a secret key for flash messages
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Required for flash messages

# Load the pre-trained model
model = joblib.load('best_model/rf_model.pkl')

# Define the ordinal encoding mappings for the categorical features
cut_mapping = {
    'Fair': 1,
    'Good': 2,
    'Very Good': 3,
    'Premium': 4,
    'Ideal': 5
}

color_mapping = {
    'D': 1,
    'E': 2,
    'F': 3,
    'G': 4,
    'H': 5,
    'I': 6,
    'J': 7
}

clarity_mapping = {
    'I1': 1,
    'SI2': 2,
    'SI1': 3,
    'VS2': 4,
    'VS1': 5,
    'VVS2': 6,
    'VVS1': 7,
    'IF': 8
}

# Initialize a StandardScaler for numerical feature scaling
scaler = StandardScaler()

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Fetch form data and validate inputs
            carat = request.form['carat']
            depth = request.form['depth']
            table = request.form['table']
            x = request.form['x']
            y = request.form['y']
            z = request.form['z']
            
            # Ensure all numerical fields have valid values
            if not all(val.replace('.', '', 1).isdigit() for val in [carat, depth, table, x, y, z]):
                flash("Please enter valid numeric values for carat, depth, table, x, y, and z.")
                return redirect(url_for('home'))
            
            carat = float(carat)
            depth = float(depth)
            table = float(table)
            x = float(x)
            y = float(y)
            z = float(z)
            
            # Ordinal encode the categorical inputs
            cut = cut_mapping[request.form['cut']]
            color = color_mapping[request.form['color']]
            clarity = clarity_mapping[request.form['clarity']]

            # Prepare the input features
            input_features = np.array([[carat, cut, color, clarity, depth, table, x, y, z]])
            
            # # Perform standard scaling on the numerical columns
            # input_features[:, [0, 4, 5, 6, 7, 8]] = scaler.fit_transform(input_features[:, [0, 4, 5, 6, 7, 8]])

            # Make the prediction using the trained model
            prediction = model.predict(input_features)
            predicted_price = round(prediction[0], 6)

            # Render the results page with the predicted price
            return render_template('results.html', final_result=predicted_price)
        
        except Exception as e:
            return render_template('results.html', final_result="Error: " + str(e))

if __name__ == '__main__':
    app.run(debug=True)
