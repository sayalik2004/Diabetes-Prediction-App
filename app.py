import streamlit as st
import pandas as pd
import joblib
import numpy as np

import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "logistic_regression_model.joblib")

model = joblib.load(model_path)
# Load the pre-trained logistic regression model
model = joblib.load('logistic_regression_model.joblib')

# Define the title of the Streamlit app
st.title('Diabetes Prediction App')
st.write('Enter the patient details to predict the likelihood of diabetes.')

# Create input fields for each feature
# Use descriptive ranges based on the dataset's descriptive statistics or common knowledge
pregnancies = st.number_input('Pregnancies', min_value=0, max_value=17, value=3, help='Number of times pregnant')
glucose = st.number_input('Glucose', min_value=44, max_value=199, value=120, help='Plasma glucose concentration a 2 hours in an oral glucose tolerance test')
blood_pressure = st.number_input('BloodPressure', min_value=24, max_value=122, value=72, help='Diastolic blood pressure (mm Hg)')
skin_thickness = st.number_input('SkinThickness', min_value=7, max_value=99, value=29, help='Triceps skin fold thickness (mm)')
insulin = st.number_input('Insulin', min_value=14, max_value=846, value=125, help='2-Hour serum insulin (mu U/ml)')
bmi = st.number_input('BMI', min_value=18.2, max_value=67.1, value=32.0, format="%.1f", help='Body Mass Index (weight in kg/(height in m)^2)')
dpf = st.number_input('DiabetesPedigreeFunction', min_value=0.078, max_value=2.42, value=0.372, format="%.3f", help='Diabetes pedigree function')
age = st.number_input('Age', min_value=21, max_value=81, value=29, help='Age (years)')

# Create a button for prediction
if st.button('Predict Outcome'):
    # Collect user inputs into a pandas DataFrame
    user_input = pd.DataFrame([{
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }])

    # Ensure column order matches training data (X_train had this order)
    # It's good practice to explicitly define column order if not directly using original X.columns
    # For this task, we can assume the order from X.columns is maintained, or load X.columns as a reference
    # For safety, let's ensure the order based on the `X` dataframe used during training
    feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    user_input = user_input[feature_names]

    # Make prediction
    prediction = model.predict(user_input)
    prediction_proba = model.predict_proba(user_input)[:, 1]

    # Display the prediction
    st.subheader('Prediction Result:')
    if prediction[0] == 1:
        st.write('<p style="font-size:20px; color:red;">The model predicts: <b>Diabetes</b></p>', unsafe_allow_html=True)
    else:
        st.write('<p style="font-size:20px; color:green;">The model predicts: <b>No Diabetes</b></p>', unsafe_allow_html=True)

    st.write(f'Probability of Diabetes: {prediction_proba[0]*100:.2f}%')

st.markdown("")
