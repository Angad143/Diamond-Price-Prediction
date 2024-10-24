from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('best_model/rf_model.pkl')

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        # Fetch form data
        carat = float(request.form['carat'])
        depth = float(request.form['depth'])
        table = float(request.form['table'])
        x = float(request.form['x'])
        y = float(request.form['y'])
        z = float(request.form['z'])
        cut = request.form['cut']
        color = request.form['color']
        clarity = request.form['clarity']
        
        # Process categorical features like cut, color, clarity (you might need encoding)
        # Assuming you have pre-processed them during model training

        # Example transformation (might differ based on how your model expects the input)
        input_features = np.array([[carat, cut, color, clarity, depth, table, x, y, z]])

        # Make prediction
        prediction = model.predict(input_features)
        predicted_price = round(prediction[0], 2)

        # Return result to the form
        return render_template('forms.html', final_result=predicted_price)

    return render_template('forms.html', final_result=None)

if __name__ == '__main__':
    app.run(debug=True)
