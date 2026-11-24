
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓"
)

st.title("🎓 Student Performance Prediction System")

st.write(
    "Enter student details to predict academic performance."
)

model_path = Path(__file__).parent / "student_model.pkl"

if not model_path.exists():
    st.error("Model not found. Please run train_model.py first.")
    st.stop()

model = joblib.load(model_path)

study_hours = st.number_input(
    "Daily Study Hours",
    min_value=0.0,
    max_value=12.0,
    value=3.0
)

attendance = st.number_input(
    "Attendance Percentage",
    min_value=0.0,
    max_value=100.0,
    value=75.0
)

previous_marks = st.number_input(
    "Previous Exam Marks",
    min_value=0.0,
    max_value=100.0,
    value=60.0
)

assignments = st.number_input(
    "Assignments Completed",
    min_value=0,
    max_value=10,
    value=5
)

if st.button("Predict Performance"):

    student_data = pd.DataFrame([{
        "study_hours": study_hours,
        "attendance": attendance,
        "previous_marks": previous_marks,
        "assignments": assignments
    }])

    prediction = model.predict(student_data)

    probability = model.predict_proba(student_data)[0][1]

    if prediction[0] == 1:
        st.success("Prediction: Likely to PASS")
    else:
        st.warning("Prediction: Needs Improvement")

    st.write(
        "Estimated pass probability:",
        f"{probability:.2%}"
    )

    st.caption(
        "Educational demonstration using synthetic data, "
        "not a validated real-world academic assessment."
    )
