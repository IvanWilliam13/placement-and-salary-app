import streamlit as st
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
import __main__
import sys
import types

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self 
    
    def transform(self, X):
        X_new = X.copy()

        # Encoding
        if 'gender' in X_new.columns:
            X_new['gender'] = X_new['gender'].map({'Male': 1, 'Female': 0})
        if 'extracurricular_activities' in X_new.columns:
            X_new['extracurricular_activities'] = X_new['extracurricular_activities'].map({'Yes': 1, 'No': 0})
        
        # Feature Engineering
        X_new['total_experience'] = (X_new['internship_count'] + X_new['live_projects'] + (X_new['work_experience_months'] / 12))
        X_new['skill_score_combined'] = (X_new['technical_skill_score'] + X_new['soft_skill_score']) / 2
        X_new['academic_score_avg'] = (X_new['ssc_percentage'] + X_new['hsc_percentage'] + X_new['degree_percentage'] + X_new['cgpa'] * 10) / 4
        X_new['engagement_score'] = (X_new['attendance_percentage'] + X_new['extracurricular_activities'])
        
        X_new.fillna(0, inplace=True)
        return X_new

train_module = types.ModuleType("train")
train_module.FeatureEngineer = FeatureEngineer
sys.modules["train"] = train_module

# Load Model.pkl
@st.cache_resource
def load_models():
    clf_model = joblib.load("artifacts/best_clf_pipeline.pkl")
    reg_model = joblib.load("artifacts/best_reg_pipeline.pkl")
    return clf_model, reg_model

clf_pipeline, reg_pipeline = load_models()

# Streamlit
st.set_page_config(page_title="Placement & Salary Prediction", layout="wide")
st.title("Student Placement & Salary Prediction")
st.markdown("""This system evaluates candidate profiles to estimate the probability of job placement.
            For candidates predicted to be placed, it also provides an estimated salary package (LPA).""")

st.sidebar.header("Candidate Profile Input")

with st.sidebar.form("prediction_form"):
   st.subheader("Demographics & Academics")
   gender = st.selectbox("Gender", ["Male", "Female"])
   ssc_percentage = st.number_input("SSC Percentage (10th Grade)", min_value=0.0, max_value=100.0, value=75.0)
   hsc_percentage = st.number_input("HSC Percentage (12th Grade)", min_value=0.0, max_value=100.0, value=75.0)
   degree_percentage = st.number_input("Undergraduate Degree Percentage", min_value=0.0, max_value=100.0, value=70.0)
   cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.5)
   entrance_exam_score = st.number_input("Entrance Exam Score", min_value=0.0, max_value=100.0, value=70.0)
   backlogs = st.number_input("Active/Past Backlogs", min_value=0, max_value=10, value=0)
   attendance_percentage = st.number_input("Attendance Percentage", min_value=0.0, max_value=100.0, value=85.0)
   extracurricular_activities = st.selectbox("Extracurricular Activities", ["Yes", "No"])
   
   st.subheader("Experience & Skills")
   internship_count = st.number_input("Total Internships", min_value=0, max_value=10, value=0)
   live_projects = st.number_input("Total Live Projects", min_value=0, max_value=10, value=0)
   work_experience_months = st.number_input("Work Experience (Months)", min_value=0, max_value=120, value=0)
   technical_skill_score = st.number_input("Technical Skill Score (0-100)", min_value=0, max_value=100, value=70)
   soft_skill_score = st.number_input("Soft Skill Score (0-100)", min_value=0, max_value=100, value=70)
   certifications = st.number_input("Total Certifications", min_value=0, max_value=10, value=0)
   
   submit_button = st.form_submit_button("Run Analysis")

# Inference Logic & Clipping (Out Of Bounds)
TRAIN_BOUNDS = {
    'ssc_percentage': (50.0, 95.0),
    'hsc_percentage': (50.0, 94.0),
    'degree_percentage': (55.0, 89.0),
    'cgpa': (5.5, 9.8),
    'entrance_exam_score': (40.0, 99.0),
    'technical_skill_score': (40.0, 99.0),
    'soft_skill_score': (40.0, 99.0),
    'internship_count': (0, 4),
    'live_projects': (0, 5),
    'work_experience_months': (0, 24),
    'certifications': (0, 5),
    'attendance_percentage': (60.0, 99.0),
    'backlogs': (0, 5)
}

if submit_button:
    input_data = pd.DataFrame({
        'gender': [gender],
        'ssc_percentage': [ssc_percentage],
        'hsc_percentage': [hsc_percentage],
        'degree_percentage': [degree_percentage],
        'cgpa': [cgpa],
        'entrance_exam_score': [entrance_exam_score],
        'technical_skill_score': [technical_skill_score],
        'soft_skill_score': [soft_skill_score],
        'internship_count': [internship_count],
        'live_projects': [live_projects],
        'work_experience_months': [work_experience_months],
        'certifications': [certifications],
        'attendance_percentage': [attendance_percentage],
        'backlogs': [backlogs],
        'extracurricular_activities': [extracurricular_activities]
    })
    
    out_of_bounds_msgs = []
    for col, (min_val, max_val) in TRAIN_BOUNDS.items():
        val = input_data[col].iloc[0]
        if val < min_val or val > max_val:
            out_of_bounds_msgs.append(f"**{col}**: {val} (Normal Range: {min_val} - {max_val})")
            
    if out_of_bounds_msgs:
        st.warning("**Extrapolation Warning:** Certain inputs exceed the historical training data range. Prediction accuracy may be affected as the model is extrapolating beyond known boundaries:\n\n" + "\n".join([f"- {msg}" for msg in out_of_bounds_msgs]))
    
    with st.spinner('Processing candidate data...'):
        placement_pred = clf_pipeline.predict(input_data)[0]
        placement_prob = clf_pipeline.predict_proba(input_data)[0][1]
        
        st.markdown("---")
        st.subheader("Analysis Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="Placement Probability", value=f"{placement_prob * 100:.1f}%")
            st.progress(float(placement_prob))
            
            if placement_pred == 1:
                st.success("Status: Placed")
            else:
                st.error("Status: Not Placed")
                
        with col2:
            if placement_pred == 1:
                reg_input = input_data.copy()
                for col, (min_val, max_val) in TRAIN_BOUNDS.items():
                    reg_input[col] = reg_input[col].clip(lower=min_val, upper=max_val)
                
                salary_pred = reg_pipeline.predict(reg_input)[0]
                
                if salary_pred < 0:
                    salary_pred = 0.0
                    
                st.metric(label="Estimated Salary (LPA)", value=f"₹ {salary_pred:.2f} Lakhs")
                st.info("Note: The salary estimate utilizes data clipping to prevent projection errors from extreme input values.")
            else:
                st.metric(label="Estimated Salary (LPA)", value="₹ 0.00 Lakhs")
                st.warning("Compensation estimate is unavailable as the candidate is not projected for placement.")
