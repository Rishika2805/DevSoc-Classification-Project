import pandas as pd
import joblib
import streamlit as st


# Page Configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Performance Predictor")
st.write("Enter the student's details below to predict their placement status.")


# Load Model
pipeline = joblib.load("./models/student_performance_model.pkl")
feature_names = joblib.load("./models/features_names.pkl")


# User Inputs
study_hours = st.slider("Study Hours", 0, 12, 6)

attendance = st.slider("Attendance (%)", 0, 100, 80)

sleep_hours = st.slider("Sleep Hours", 0, 12, 7)

internet_usage = st.slider("Internet Usage (Hours)", 0, 12, 3)

assignments_completed = st.slider("Assignments Completed", 0, 10, 8)

previous_score = st.slider("Previous Score", 0, 100, 75)

exam_score = st.slider("Exam Score", 0, 100, 80)

# Prediction
if st.button("Predict"):

    input_data = pd.DataFrame(
        [[
            study_hours,
            attendance,
            sleep_hours,
            internet_usage,
            assignments_completed,
            previous_score,
            exam_score
        ]],
        columns=feature_names
    )

    prediction = pipeline.predict(input_data)[0]

    if prediction == 1:
        st.success(
            f"🎉 Student is likely to be **Placed!**\n\n"
        )
    else:
        st.error(
            f"❌ Student is **Not Likely to be Placed.**\n\n"
        )