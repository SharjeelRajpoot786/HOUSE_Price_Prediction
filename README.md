# HOUSE_Price_Prediction
In this Project we are using the large data set of Test and train and train and test the model and deploy it by using Flask 
Flask App for House Price Prediction Code :
from flask import Flask, request, render_template
import joblib
app = Flask(__name__)
# Load model and columns
model = joblib.load("model.pkl")
model_columns = joblib.load("model_columns.pkl")
@app.route('/')
def home():
    print("Rendering home page...")
    return render_template('index.html', model_columns=model_columns)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = [float(request.form[col]) for col in model_columns]
        prediction = model.predict([input_data])[0]
        return render_template('index.html',
                               prediction_text=f"Predicted House Price: ${prediction:,.2f}",
                               model_columns=model_columns)
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(debug=True)

1. Flask App for House Price Prediction
This part is for the web interface:
•	Flask creates a small website.
•	The model.pkl (your trained model) and model_columns.pkl (the features your model expects) are loaded.
•	The home page (/) shows a form with input fields (based on model_columns).
•	When the form is submitted (to /predict), the app:
o	Reads the input values.
o	Predicts the price using your model.
o	Shows the result on the same page.
✅ Purpose: To let users enter house details and get a price prediction on the web.
________________________________________
📊 2. Testing the Model on Test Data Code:
import pandas as pd
import numpy as np
import joblib
# Load test data
test_data = pd.read_csv("/content/drive/MyDrive/HousePrice/test.csv")
# Load model and columns
model = joblib.load("/content/drive/MyDrive/HousePrice/model.pkl")
model_columns = joblib.load("/content/drive/MyDrive/HousePrice/model_columns.pkl")
# Select only numeric and handle missing columns
test_data_num = test_data.select_dtypes(include=[np.number])
# Fill missing and add missing columns
for col in model_columns:
    if col not in test_data_num.columns:
        test_data_num[col] = 0
    test_data_num[col].fillna(test_data_num[col].median(), inplace=True)
# Match column order
X_test = test_data_num[model_columns]
# Predict
predictions = model.predict(X_test)
# Show result as DataFrame
output = pd.DataFrame({'Id': test_data['Id'], 'SalePrice': predictions})
print(output.head(10))  # Show first 10 rows; change or remove for full output
📊 2. Testing the Model on Test Data
This part is for making predictions on a CSV file:
•	Loads a test CSV file.
•	Loads the trained model and expected column names.
•	Makes sure the test data:
o	Has the right columns (adds any missing ones).
o	Fills in any missing values.
•	Reorders the columns to match what the model expects.
•	Uses the model to predict prices for each row.
•	Prints the first 10 predictions (with house Id and SalePrice).
✅ Purpose: To run the model on actual test data and see predicted house prices.
________________________________________
📈 3. Training the Model Code :
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
# Load the dataset
data = pd.read_csv("/content/drive/MyDrive/HousePrice/train.csv")
data.head()
📈 3. Training the Model:
This part is for training your prediction model:
•	Loads the training dataset.
•	(Your code stops at data.head(), but normally you would:)
o	Clean the data.
o	Choose input features and the target (house price).
o	Split the data into training/testing sets.
o	Train a model (like RandomForestRegressor).
o	Save it using joblib (as model.pkl).

