import streamlit as st
import requests

API_URL = "https://diabetes-1-c89w.onrender.com/predict"

st.set_page_config(page_title="Diabetes Probability Predictor", layout="centered")
st.title("🩺 Diabetes Probability Predictor")

# -------------------- Inputs --------------------
age = st.number_input("Age", min_value=1, max_value=119, value=25)

alcohol_consumption_per_week = st.slider(
    "Alcohol consumption per week", 0.0, 10.0, 0.0
)

physical_activity_minutes_per_week = st.slider(
    "Physical activity (minutes/week)", 0, 800, 300
)

diet_score = st.slider("Diet score (0–10)", 0.0, 10.0, 7.0)
sleep_hours_per_day = st.slider("Sleep hours per day", 0.0, 24.0, 8.0)
screen_time_hours_per_day = st.slider("Screen time (hours/day)", 0.0, 20.0, 4.0)

# -------------------- Height & Weight --------------------
st.subheader("Height & Weight")

height_value = st.number_input("Height value", min_value=0.1, value=170.0)
height_unit = st.selectbox("Height unit", ["cm", "m", "ft"])
weight_kg = st.number_input("Weight (kg)", min_value=1.0, value=70.0)

# -------------------- Medical --------------------
waist_to_hip_ratio = st.slider("Waist-to-hip ratio", 0.1, 1.0, 0.85)

systolic_bp = st.number_input("Systolic BP", value=120)
diastolic_bp = st.number_input("Diastolic BP", value=80)
heart_rate = st.slider("Heart rate", 40, 100, 70)

cholesterol_total = st.slider("Total cholesterol", 100, 300, 180)
hdl_cholesterol = st.slider("HDL cholesterol", 20, 100, 55)
ldl_cholesterol = st.slider("LDL cholesterol", 50, 210, 100)
triglycerides = st.slider("Triglycerides", 90, 300, 150)

# -------------------- History --------------------
family_history_diabetes = st.selectbox("Family history of diabetes", ["no", "yes"])
hypertension_history = st.selectbox("Hypertension history", ["no", "yes"])
cardiovascular_history = st.selectbox("Cardiovascular history", ["no", "yes"])

# -------------------- Categorical --------------------
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
ethnicity = st.selectbox("Ethnicity", ["Hispanic", "White", "Asian", "Black", "Other"])
education_level = st.selectbox(
    "Education level", ["Highschool", "Graduate", "Postgraduate", "No formal"]
)
income_level = st.selectbox(
    "Income level", ["Low", "Lower-Middle", "Middle", "Upper-Middle", "High"]
)
smoking_status = st.selectbox("Smoking status", ["Never", "Former", "Current"])
employment_status = st.selectbox(
    "Employment status", ["Employed", "Student", "Retired", "Unemployed"]
)

# -------------------- Predict --------------------
if st.button("Predict Diabetes Probability"):
    payload = {
        "age": age,
        "alcohol_consumption_per_week": alcohol_consumption_per_week,
        "physical_activity_minutes_per_week": physical_activity_minutes_per_week,
        "diet_score": diet_score,
        "sleep_hours_per_day": sleep_hours_per_day,
        "screen_time_hours_per_day": screen_time_hours_per_day,
        "height_value": height_value,
        "height_unit": height_unit,
        "weight_kg": weight_kg,
        "waist_to_hip_ratio": waist_to_hip_ratio,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "heart_rate": heart_rate,
        "cholesterol_total": cholesterol_total,
        "hdl_cholesterol": hdl_cholesterol,
        "ldl_cholesterol": ldl_cholesterol,
        "triglycerides": triglycerides,
        "gender": gender,
        "ethnicity": ethnicity,
        "education_level": education_level,
        "income_level": income_level,
        "smoking_status": smoking_status,
        "employment_status": employment_status,
        "family_history_diabetes": family_history_diabetes,
        "hypertension_history": hypertension_history,
        "cardiovascular_history": cardiovascular_history,
    }

    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            prob = response.json()["probability"]
            st.success(f"🧠 Diabetes Probability: **{prob * 100:.2f}%**")
        else:
            st.error(f"API Error: {response.text}")
    except Exception as e:
        st.error(f"Connection error: {e}")
