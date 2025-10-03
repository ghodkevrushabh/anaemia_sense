# Anemia Sense: AI-Powered Anemia Risk Prediction 🩸

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Framework](https://img.shields.io/badge/Flask-2.0-black?style=for-the-badge&logo=flask)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

A powerful and user-friendly web application that leverages a Gradient Boosting Classifier to predict the risk of anemia based on key blood parameters. This project was developed as a comprehensive machine learning solution, from data analysis to model deployment, achieving **100% accuracy** on the test set.

![Hero Image](assets/hero-image.png)

## Table of Contents
- [Problem Statement](#problem-statement)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Getting Started](#getting-started)
- [How to Use the Model](#how-to-use-the-model)
- [Contributing](#contributing)
- [License](#license)

## Problem Statement
Anemia is a significant global health issue affecting millions. Early detection is crucial for timely and effective treatment. This project aims to build a reliable predictive tool that can assist in the preliminary screening of anemia by analyzing standard blood test results, making the process faster and more accessible.

## Key Features
- **High-Accuracy Predictions**: Utilizes a Gradient Boosting Classifier that achieved 100% accuracy during testing.
- **Comprehensive EDA**: In-depth Exploratory Data Analysis with rich visualizations to understand the data's story.
- **Class Imbalance Handling**: Employs downsampling to create a balanced and robust dataset, leading to a more reliable model.
- **Multi-Algorithm Evaluation**: Compares 6 different classification models to ensure the best one is chosen for the task.
- **Interactive Web Interface**: A user-friendly web app (built with Flask) to input health parameters and get instant predictions.
- **Model Persistence**: The trained model is saved using `pickle` for easy deployment and reuse.

## System Architecture
The project follows a standard machine learning workflow, from data preprocessing and model training to final prediction via a user interface.

![Architecture Diagram](assets/architecture.png)

## Tech Stack
- **Backend**: Python, Flask
- **Data Science**: Pandas, NumPy, Scikit-learn
- **Data Visualization**: Matplotlib, Seaborn
- **Frontend**: HTML, CSS

## Dataset
The model is trained on the `anemia.csv` dataset. The features used for prediction include:

| Feature      | Description                                | Type  |
|--------------|--------------------------------------------|-------|
| Gender       | 0 for Male, 1 for Female                   | Int   |
| Hemoglobin   | Hemoglobin level (g/dL)                    | Float |
| MCH          | Mean Corpuscular Hemoglobin (pg)           | Float |
| MCHC         | Mean Corpuscular Hemoglobin Concentration (%) | Float |
| MCV          | Mean Corpuscular Volume (fL)               | Float |
| **Result (Target)** | **0 for Not Anemic, 1 for Anemic** | **Int** |

## Getting Started
To get a local copy up and running, follow these simple steps.

### Prerequisites
- Python 3.8+
- `pip` package manager

### Installation
1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/ghodkevrushabh/anaemia_sense.git](https://github.com/ghodkevrushabh/anaemia_sense.git)
    cd anaemia_sense
    ```

2.  **Create and activate a virtual environment (recommended):**
    - **Windows:**
      ```sh
      python -m venv venv
      .\venv\Scripts\activate
      ```
    - **macOS/Linux:**
      ```sh
      python3 -m venv venv
      source venv/bin/activate
      ```

3.  **Install the required packages:**
    ```sh
    pip install flask pandas numpy scikit-learn
    ```

4.  **Run the Flask application:**
    ```sh
    python app.py
    ```

5.  Open your browser and navigate to `http://127.0.0.1:5000`.

## How to Use the Model
Besides the web interface, you can use the saved `model.pkl` file to make predictions in any Python script.

```python
import pickle
import pandas as pd

# Load the saved model
with open('model(2).pkl', 'rb') as file:
    model = pickle.load(file)

# Define feature names in the correct order
feature_names = ['Gender', 'Hemoglobin', 'MCH', 'MCHC', 'MCV']

# Create a DataFrame for new data (Example: a male with these parameters)
# Gender=0 (Male), HGB=11.6, MCH=22.3, MCHC=30.9, MCV=74.5
new_data = pd.DataFrame([[0, 11.6, 22.3, 30.9, 74.5]], columns=feature_names)

# Make a prediction
prediction = model.predict(new_data)

# Interpret the result
if prediction[0] == 0:
    print("Prediction: You are likely NOT Anemic.")
else:
    print("Prediction: You have a high risk of being Anemic.")
