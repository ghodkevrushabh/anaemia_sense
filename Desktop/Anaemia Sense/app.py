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
    return render_template('result.html', show_result=False)

# Define the route for prediction
@app.route('/predict', methods=["POST"])
def predict():
    try:
        # Get the input values from the form
        gender = float(request.form["gender"])
        hemoglobin = float(request.form["hemoglobin"])
        mch = float(request.form["mch"])
        mchc = float(request.form["mchc"])
        mcv = float(request.form["mcv"])

        # Validate input ranges
        if gender not in [0, 1]:
            return render_template('result.html', 
                                 show_result=True,
                                 result_title="Invalid Input",
                                 result_message="Please select a valid gender.",
                                 result_icon="⚠️",
                                 result_advice="Please go back and fill out the form correctly.")

        # Create a numpy array for the model
        features_values = np.array([[gender, hemoglobin, mch, mchc, mcv]])
        
        # Create a pandas DataFrame with correct feature names
        df = pd.DataFrame(features_values, columns=['Gender', 'Hemoglobin', 'MCH', 'MCHC', 'MCV'])

        # Get the prediction from the model
        prediction = model.predict(df)

        # Determine the result text and additional info
        if prediction[0] == 1:
            result_title = "Positive for Anemia"
            result_message = "Based on your health parameters, our AI model indicates potential anemic condition."
            result_icon = "⚠️"
            result_advice = "We strongly recommend consulting with a healthcare professional immediately for proper diagnosis and treatment. Early intervention can help manage the condition effectively. Consider getting a complete blood count (CBC) test and follow your doctor's recommendations for treatment options including iron supplements, dietary changes, or other medical interventions."
        else:
            result_title = "Negative for Anemia"
            result_message = "Based on your health parameters, our AI model indicates no signs of anemic condition."
            result_icon = "✅"
            result_advice = "Your results look promising! Continue maintaining a healthy lifestyle with a balanced diet rich in iron, vitamin B12, and folate. Include foods like lean meats, leafy greens, legumes, and fortified cereals. Regular check-ups are still recommended to monitor your health. Keep up with proper nutrition, regular exercise, and adequate rest for optimal wellbeing."
        
        return render_template('result.html', 
                             show_result=True,
                             result_title=result_title,
                             result_message=result_message,
                             result_icon=result_icon,
                             result_advice=result_advice)
    
    except ValueError:
        return render_template('result.html', 
                             show_result=True,
                             result_title="Input Error",
                             result_message="Please enter valid numeric values for all health parameters.",
                             result_icon="❌",
                             result_advice="Check that all fields contain proper numbers and try again. If you're unsure about your values, please consult your recent blood test results or contact your healthcare provider.")
    
    except Exception as e:
        return render_template('result.html', 
                             show_result=True,
                             result_title="Processing Error",
                             result_message="An unexpected error occurred while processing your request.",
                             result_icon="❌",
                             result_advice="Please try again. If the problem persists, contact support. This tool is for informational purposes only - always consult a healthcare professional for medical advice.")

# Main function to run the application
if __name__ == "__main__":
    app.run(debug=True)