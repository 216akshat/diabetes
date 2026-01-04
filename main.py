from pydantic import BaseModel, Field, field_validator,computed_field
from typing import Literal, Optional
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import pickle
import pandas as pd
# -------------------- Load ML Model --------------------
with open("diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)
# -------------------- FastAPI App --------------------
app = FastAPI(title="Diabetes Probability API")




class DiabetesInput(BaseModel):
    # -------------------- Demographics --------------------
    age: int = Field(..., gt=0, lt=120, description="Age in years")
    alcohol_consumption_per_week: float = Field(..., ge=0, le=10, description="Alcoholic drinks per week")
    physical_activity_minutes_per_week: float = Field( ..., ge=0, le=800, description="Weekly physical activity in minutes" )
    diet_score: float = Field( ..., ge=0, le=10, description="Diet quality score (0–10)" )
    sleep_hours_per_day: float = Field( ..., ge=0, le=24, description="Average sleep hours per day" )
    screen_time_hours_per_day: float = Field( ..., ge=0, le=20, description="Daily screen time in hours" )

    height_value: float = Field(..., gt=0, description="Height value")
    height_unit: Literal["m", "cm", "ft"] = Field( ..., description="Height unit: meters (m), centimeters (cm), feet (ft)" )
    weight_kg: float = Field(..., gt=0, description="Weight in kilograms")

    @computed_field
    @property
    def bmi(self)-> float:
        hv = self.height_value
        hu = self.height_unit
        w = self.weight_kg

        if hu == "cm":
            h = hv / 100
        elif hu == "ft":
            h = hv * 0.3048
        else:
            h = hv 
        return w/(h**2)
    
    waist_to_hip_ratio: float = Field(..., gt=0, le=1)
    systolic_bp: int
    diastolic_bp: int
    heart_rate: int = Field(..., ge=40, le=100)

    cholesterol_total: int = Field(..., ge=100, le=300)
    hdl_cholesterol: int = Field(..., ge=20, le=100)
    ldl_cholesterol: int = Field(..., ge=50, le=210)
    triglycerides: int = Field(..., ge=90, le=300)

    family_history_diabetes: int
    hypertension_history: int
    cardiovascular_history: int


    @field_validator(
        "family_history_diabetes",
        "hypertension_history",
        "cardiovascular_history",
        mode="before"
    )
    @classmethod
    def yes_no_to_binary(cls, v):
        return 1 if v.lower() == "yes" else 0

    gender: Literal["Male", "Female", "Other"]
    ethnicity: Literal["Hispanic", "White", "Asian", "Black", "Other"]
    education_level: Literal["Highschool", "Graduate", "Postgraduate", "No formal"]
    income_level: Literal["Lower-Middle", "Upper-Middle", "Low", "Middle", "High"]
    smoking_status: Literal["Current", "Never", "Former"]
    employment_status: Literal["Employed", "Retired", "Student", "Unemployed"]
@app.post("/predict")
def predict(data: DiabetesInput):

    df = pd.DataFrame([{
        "age": data.age,
        "alcohol_consumption_per_week": data.alcohol_consumption_per_week,
        "physical_activity_minutes_per_week": data.physical_activity_minutes_per_week,
        "diet_score": data.diet_score,
        "sleep_hours_per_day": data.sleep_hours_per_day,
        "screen_time_hours_per_day": data.screen_time_hours_per_day,
        "bmi": data.bmi,
        "waist_to_hip_ratio": data.waist_to_hip_ratio,
        "systolic_bp": data.systolic_bp,
        "diastolic_bp": data.diastolic_bp,
        "heart_rate": data.heart_rate,
        "cholesterol_total": data.cholesterol_total,
        "hdl_cholesterol": data.hdl_cholesterol,
        "ldl_cholesterol": data.ldl_cholesterol,
        "triglycerides": data.triglycerides,
        "gender": data.gender,
        "ethnicity": data.ethnicity,
        "education_level": data.education_level,
        "income_level": data.income_level,
        "smoking_status": data.smoking_status,
        "employment_status": data.employment_status,
        "family_history_diabetes": data.family_history_diabetes,
        "hypertension_history": data.hypertension_history,
        "cardiovascular_history": data.cardiovascular_history
    }])

    prob = model.predict_proba(df)[0][1]

    return {"probability": round(float(prob), 4)}

 
