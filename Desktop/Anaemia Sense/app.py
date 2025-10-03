import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, render_template

# Initialize the Flask application
app = Flask(__name__)

# Load the saved machine learning model
model = pickle.load(open('model(2).pkl', 'rb'))

# Define the route for the home page (index.html)
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for the details/about page (details.html)
@app.route('/details')
def details():
    return render_template('details.html')

# Define the route for the result/prediction form page (result.html)
@app.route('/result')
def result():
    return render_template('result.html')

# Define the route for prediction
@app.route('/predict', methods=["POST"])
def predict():
    # Get the input values from the form
    gender = float(request.form["gender"])  # Changed to lowercase to match HTML
    hemoglobin = float(request.form["hemoglobin"])  # Changed to lowercase
    mch = float(request.form["mch"])  # Changed to lowercase
    mchc = float(request.form["mchc"])  # Changed to lowercase
    mcv = float(request.form["mcv"])  # Changed to lowercase

    # Create a numpy array for the model
    features_values = np.array([[gender, hemoglobin, mch, mchc, mcv]])
    
    # Create a pandas DataFrame with correct feature names
    df = pd.DataFrame(features_values, columns=['Gender', 'Hemoglobin', 'MCH', 'MCHC', 'MCV'])

    # Get the prediction from the model
    prediction = model.predict(df)

    # Determine the result text and additional info
    if prediction[0] == 1:
        result_title = "Positive for Anemia"
        result_message = "Based on your health parameters, you have anemic disease."
        result_icon = "⚠️"
        result_advice = "Please consult with a healthcare professional immediately for proper diagnosis and treatment. Early intervention can help manage the condition effectively. Consider getting a complete blood count test and follow your doctor's recommendations."
    else:
        result_title = "Negative for Anemia"
        result_message = "Based on your health parameters, you don't have any anemic disease."
        result_icon = "✅"
        result_advice = "Your results look good! Continue maintaining a healthy diet rich in iron, vitamin B12, and folate. Regular check-ups are still recommended to monitor your health. Keep up with a balanced lifestyle including proper nutrition and exercise."
    
    return render_template('result.html', 
                         show_result=True,
                         result_title=result_title,
                         result_message=result_message,
                         result_icon=result_icon,
                         result_advice=result_advice)

# Main function to run the application
if __name__ == "__main__":
    app.run(debug=True)