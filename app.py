import streamlit as st
import pandas as pd
import pickle
import sklearn

# Load the saved model
with open('depression_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Function to predict depression
def predict_depression(data):
    prediction = model.predict(data)
    return prediction

# Streamlit app
st.title("Depression Prediction App ")

# Input features
with st.form("prediction_form"):
    gender = st.selectbox("Gender", ["Female", "Male"])
    gender = 0 if gender == "Female" else 1

    age = st.number_input("Age", min_value=0, max_value=100, value=20)

    academic_pressure = st.number_input("Academic Pressure", min_value=0.0, max_value=5.0, value=5.0, step=0.1)
    cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
    study_satisfaction = st.number_input("Study Satisfaction", min_value=0.0, max_value=5.0, value=5.0, step=0.1)

    dietary_habits = st.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy"])
    dietary_habits = 0 if dietary_habits == "Healthy" else (1 if dietary_habits == "Moderate" else 2)

    suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts?", ["No", "Yes"])
    suicidal_thoughts = 0 if suicidal_thoughts == "No" else 1

    work_study_hours = st.number_input("Work/Study Hours", min_value=0.0, max_value=24.0, value=8.0, step=0.1)
    financial_stress = st.number_input("Financial Stress", min_value=0.0, max_value=5.0, value=5.0, step=0.1)

    family_history = st.selectbox("Family History of Mental Illness", ["No", "Yes"])
    family_history = 0 if family_history == "No" else 1

    sleep_duration = st.selectbox("Sleep Duration", ["More than 8 hours", "7-8 hours", "5-6 hours", "Less than 5 hours"])
    sleep_duration_encoded = 1 if sleep_duration == "More than 8 hours" else (2 if sleep_duration == "7-8 hours" else (3 if sleep_duration == "5-6 hours" else 4))

    submit_button = st.form_submit_button(label='Predict')

# Create a DataFrame from the input features
if submit_button:
    data = pd.DataFrame({
        'Gender': [gender],
        'Age': [age],
        'Academic Pressure': [academic_pressure],
        'CGPA': [cgpa],
        'Study Satisfaction': [study_satisfaction],
        'Dietary Habits': [dietary_habits],
        'Have you ever had suicidal thoughts ?': [suicidal_thoughts],
        'Work/Study Hours': [work_study_hours],
        'Financial Stress': [financial_stress],
        'Family History of Mental Illness': [family_history],
        'sleep_duration_encoded': [sleep_duration_encoded]
    })

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    data[['Age', 'Academic Pressure', 'CGPA', 'Study Satisfaction', 'Work/Study Hours', 'Financial Stress']] = scaler.fit_transform(data[['Age', 'Academic Pressure', 'CGPA', 'Study Satisfaction', 'Work/Study Hours', 'Financial Stress']])

    prediction = predict_depression(data)
    if prediction[0] == 0:
        st.write("Prediction: Not Depressed")
    else:
        st.write("Prediction: Likely Depressed")