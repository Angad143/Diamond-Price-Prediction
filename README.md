
# Diamonds Price Prediction 💎

This project predicts diamond prices based on features like carat, cut, color, and clarity. Leveraging Exploratory Data Analysis (EDA) and a Random Forest Regression model, this project also includes a user-friendly web application built with Flask, allowing users to input diamond features and receive instant price predictions.

## 📜 Project Overview
The Diamonds Price Prediction model uses machine learning techniques to provide an accurate price estimation based on a diamond's characteristics. Starting with data preprocessing and EDA, this project guides through feature transformations, model training, and evaluation using metrics like Mean Absolute Error (MAE). The best-performing model is integrated into a Flask web application for easy user interaction.

## 📂 Project Steps

### Step 1: Import Necessary Libraries and Load the Dataset
The project begins by loading required Python libraries and importing the dataset.

### Step 2: Exploratory Data Analysis (EDA)
EDA is conducted to understand the dataset's structure, analyze relationships, and prepare features for modeling.

- **Univariate Analysis**  
  - *Numerical Columns*: Carat, Depth, Table, X, Y, Z  
  - *Categorical Columns*: Cut, Color, Clarity  

- **Bivariate Analysis**  
  - *Price vs Numerical Columns*: Carat, Depth, Table, X, Y, Z  
  - *Price vs Categorical Columns*: Cut, Color, Clarity  

### Step 3: Data Preparation & Cleaning
Data cleaning ensures the dataset is ready for modeling by addressing missing values, data types, duplicates, and outliers.

1. **Handling Missing Values**: Impute or remove rows based on relevance.
2. **Change Data Types**: Convert types as needed (e.g., strings to dates).
3. **Drop Duplicates**: Eliminate redundant rows.
4. **Outlier Detection and Removal**: Address extreme values to improve model performance.

### Step 4: Train-Test Split
Split the data into training and testing sets, segregating input (X) and target (y) variables.

### Step 5: Feature Transformation (Apply on both X_train and X_test datasets)
Apply transformations to ensure the data is in an optimal format for the model.

- **Numerical Feature Transformation**: 
  - Standardization (used in this project)

- **Categorical Feature Transformation**:
  - Ordinal Encoding (used for ordered variables like cut, color, clarity)

> *Note*: While Ordinal Encoding is used here due to inherent order in features, other techniques like One-Hot Encoding are ideal for unordered categorical variables or distance-based models like KNN.

### Step 6: Model Training & Evaluation
Train and evaluate multiple models to identify the best-performing one.

- **Models Used**: K-Nearest Neighbors (KNN) Regression, Linear Regression, Decision Tree regression and Random Forest Regression
- **Best Model**: Random Forest Regression with the lowest Mean Absolute Error (MAE) of 233.21
- **Evaluation Metrics**:
  - Train and Test set predictions, visualized and evaluated for accuracy using MAE.

### Step 7: Building the Flask Web Application
To make the model accessible and user-friendly, a Flask-based web application was developed. This application allows users to input diamond attributes and receive real-time price predictions.

1. **User Interface**: The app provides a simple form to enter diamond details (carat, cut, color, clarity, etc.).
2. **Backend**: Uses the trained Random Forest model to predict the diamond price based on user input.
3. **Deployment**: Flask makes it easy to deploy the app on local servers or cloud platforms.

## 💻 Installation & Usage

### Requirements
To run this project locally, clone the repository and install the necessary packages.

```bash
git clone https://github.com/Angad143/Diamond-Price-Prediction.git
cd Diamonds_Price_Prediction
```

### Running the Web Application
1. **Start the Flask app**:
   ```bash
   python app.py
   ```
2. **Access the Application**: Open your browser and navigate to `http://127.0.0.1:5000/`.

### Running the Model in Notebook
For in-depth analysis, open `Diamonds_price_predictions_Final_Solutions.ipynb` to run the model directly in a Jupyter Notebook.

## 🔍 Results
The Random Forest Regressor proved to be the best among models, yielding the lowest MAE error of 233.21. The Flask web application integrates this model, enabling real-time price predictions based on diamond attributes.

