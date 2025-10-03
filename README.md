Anemia Sense 🩸: AI-Powered Anemia Risk Prediction
Anemia Sense is a powerful web application that leverages a Gradient Boosting Classifier to predict the risk of anemia based on key blood parameters. This tool was developed as a comprehensive machine learning project, from data analysis to model deployment, achieving 100% accuracy on the test set.

[][python-link]
[][license-link]
[][ml-link]
[][flask-link]

📋 Table of Contents
Problem Statement

Key Features

System Architecture

Illustration

Tech Stack

Dataset

Getting Started

Methodology

Model Performance

How to Use

Contributing

License

🎯 Problem Statement
Anemia is a significant global health issue affecting millions. Early detection is crucial for timely and effective treatment. This project aims to build a reliable predictive tool that can assist in the preliminary screening of anemia by analyzing standard blood test results, making the process faster and more accessible.

✨ Key Features
High-Accuracy Predictions: Utilizes a Gradient Boosting Classifier that achieved 100% accuracy during testing.

Comprehensive EDA: In-depth Exploratory Data Analysis with rich visualizations to understand the data's story.

Class Imbalance Handling: Employs downsampling to create a balanced and robust dataset, leading to a more reliable model.

Multi-Algorithm Evaluation: Compares 6 different classification models to ensure the best one is chosen for the task.

Interactive Web Interface: A user-friendly web app (built with Flask) to input health parameters and get instant predictions.

Model Persistence: The trained model is saved using pickle for easy deployment and reuse.

🏗️ System Architecture
The project follows a standard machine learning workflow, from data preprocessing and model training to final prediction via a user interface.

🩸 Illustration
This illustration visually represents the difference in red blood cell concentration between a normal state and an anemic state.

🛠️ Tech Stack
Core Language: Python

Data Manipulation: Pandas, NumPy

Data Visualization: Matplotlib, Seaborn

Machine Learning: Scikit-learn

Web Framework: Flask

Development Environment: Jupyter Notebook

Model Serialization: Pickle

📊 Dataset
The model is trained on the anemia.csv dataset. The features used for prediction include:

Feature	Description	Type
Gender	0 for Male, 1 for Female	Integer
Hemoglobin	Hemoglobin level (g/dL)	Float
MCH	Mean Corpuscular Hemoglobin (pg)	Float
MCHC	Mean Corpuscular Hemoglobin Concentration (%)	Float
MCV	Mean Corpuscular Volume (fL)	Float
Result	Target: 0 for Not Anemic, 1 for Anemic	Integer

Export to Sheets
🚀 Getting Started
To get a local copy up and running, follow these simple steps.

Prerequisites
Python 3.8+

pip package manager

Installation
Clone the repository:


git clone https://github.com/your-username/anaemia-sense.git
cd anaemia-sense
Create and activate a virtual environment (recommended):



# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
Install the required packages:



pip install pandas numpy scikit-learn matplotlib seaborn flask
Run the Flask application:



python app.py
Open your browser and navigate to http://127.0.0.1:5000.

🔬 Methodology
Data Loading & Inspection: The anemia.csv dataset was loaded and inspected for null values and data types.

Handling Class Imbalance: The dataset showed an imbalance between anemic and non-anemic cases. Downsampling of the majority class was performed to create a balanced dataset.

Exploratory Data Analysis (EDA): Univariate, bivariate, and multivariate analyses were conducted using histograms, bar plots, pair plots, and a correlation heatmap.

Model Training & Evaluation: The data was split into training (80%) and testing (20%) sets. Six different classification algorithms were trained and evaluated.

Model Selection: The Gradient Boosting Classifier was selected as the best model due to its perfect performance on the test set.

Model Saving: The final trained model was serialized and saved as model.pkl for use in the Flask web application.

📈 Model Performance
The performance of all evaluated models on the test set is summarized below. The Gradient Boosting model was chosen for its perfect accuracy and robustness.

Model	Test Set Score
Logistic Regression	0.991935
Decision Tree Classifier	1.000000
RandomForest Classifier	1.000000
Gaussian Naive Bayes	0.979839
Support Vector Classifier	0.939516
Gradient Boost Classifier	1.000000

Export to Sheets
💡 How to Use
The saved model.pkl file can be used to make predictions on new data. Here's a code snippet demonstrating how:


import pickle
import pandas as pd

# Load the saved model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define feature names in the correct order
feature_names = ['Gender', 'Hemoglobin', 'MCH', 'MCHC', 'MCV']

# Create a DataFrame for new data (Example: a male with these parameters)
new_data = pd.DataFrame([[0, 11.6, 22.3, 30.9, 74.5]], columns=feature_names)

# Make a prediction
prediction = model.predict(new_data)

# Interpret the result
if prediction[0] == 0:
    print("Prediction: You are likely NOT Anemic.")
else:
    print("Prediction: You have a high risk of being Anemic.")
🤝 Contributing
Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

