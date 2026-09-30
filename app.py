import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="EduPredict AI", page_icon="🎓", layout="wide")
st.title("🎓 EduPredict AI")
st.caption("A machine-learning demo for exploring factors associated with student outcomes.")

@st.cache_resource
def load_model():
    return joblib.load("models/student_model.joblib")

st.sidebar.header("Student details")
attendance = st.sidebar.slider("Attendance (%)", 0, 100, 80)
study_hours = st.sidebar.slider("Study hours per day", 0.0, 12.0, 3.0, 0.5)
assignment = st.sidebar.slider("Assignment score (%)", 0, 100, 70)
internal1 = st.sidebar.slider("Internal exam 1 (%)", 0, 100, 65)
internal2 = st.sidebar.slider("Internal exam 2 (%)", 0, 100, 68)
participation = st.sidebar.slider("Class participation (1–5)", 1, 5, 3)
sleep = st.sidebar.slider("Sleep hours per night", 0.0, 12.0, 7.0, 0.5)

row = pd.DataFrame([{
    "attendance_percent": attendance,
    "study_hours_per_day": study_hours,
    "assignment_score": assignment,
    "internal_exam1": internal1,
    "internal_exam2": internal2,
    "class_participation": participation,
    "sleep_hours": sleep
}])

st.subheader("Prediction")
try:
    model = load_model()
    predicted_score = float(model.predict(row)[0])
    predicted_score = max(0, min(100, predicted_score))
    a, b = st.columns(2)
    a.metric("Estimated final score", f"{predicted_score:.1f}/100")
    b.metric("Estimated outcome", "Pass" if predicted_score >= 40 else "At risk")
    st.info("This is an educational demo, not a validated tool for making decisions about students.")
except FileNotFoundError:
    st.warning("Model not found. Run `python train.py` first to create models/student_model.joblib.")

st.subheader("Your inputs")
st.dataframe(row, use_container_width=True, hide_index=True)
