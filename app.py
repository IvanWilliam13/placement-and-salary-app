import streamlit as st
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go 
import sys
import __main__
sys.modules['train'] = __main__

# Customized Transformer (required to load .pkl)
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self 
    
    def transform(self, X):
        X_new = X.copy()
        if 'gender' in X_new.columns:
            X_new['gender'] = X_new['gender'].map({'Male': 1, 'Female': 0})
        if 'extracurricular_activities' in X_new.columns:
            X_new['extracurricular_activities'] = X_new['extracurricular_activities'].map({'Yes': 1, 'No': 0})
            
        X_new['total_experience'] = (X_new['internship_count'] + X_new['live_projects'] + (X_new['work_experience_months'] / 12))
        X_new['skill_score_combined'] = (X_new['technical_skill_score'] + X_new['soft_skill_score']) / 2
        X_new['academic_score_avg'] = (X_new['ssc_percentage'] + X_new['hsc_percentage'] + X_new['degree_percentage'] + X_new['cgpa'] * 10) / 4
        X_new['engagement_score'] = (X_new['attendance_percentage'] + X_new['extracurricular_activities'])
        X_new.fillna(0, inplace=True)
        return X_new

# Load Model.pkl
@st.cache_resource
def load_models():
    clf_model = joblib.load("artifacts/best_clf_pipeline.pkl")
    reg_model = joblib.load("artifacts/best_reg_pipeline.pkl")
    return clf_model, reg_model

clf_pipeline, reg_pipeline = load_models()

# Page Configuration
st.set_page_config(page_title="Placement & Salary Predictor", page_icon="🎓", layout="wide", initial_sidebar_state="expanded")
st.title("🎓 Student Placement & Salary Predictor")
st.markdown("Predictive analysis for job placement probability and salary estimation.")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Professional Experience")
    internship_count = st.slider("Total Internships", 0, 10, 0)
    live_projects = st.slider("Total Live Projects", 0, 10, 0)
    work_experience_months = st.slider("Work Experience (Months)", 0, 120, 0)
    certifications = st.slider("Total Certifications", 0, 10, 0)

# Input Form
with st.form("prediction_form"):
    st.subheader("Demographics")
    gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
    
    st.subheader("Academics")
    extracurricular_activities_bool = st.checkbox("Active in Extracurricular Activities")
    extracurricular_activities = "Yes" if extracurricular_activities_bool else "No"
    
    col1, col2, col3 = st.columns(3)
    with col1:
        ssc_percentage = st.number_input("SSC Percentage (10th)", 0.0, 100.0, 75.0)
        hsc_percentage = st.number_input("HSC Percentage (12th)", 0.0, 100.0, 75.0)
        attendance_percentage = st.number_input("Attendance Percentage (%)", 0.0, 100.0, 85.0)
    with col2:
        degree_percentage = st.number_input("Degree Percentage", 0.0, 100.0, 70.0)
        cgpa = st.number_input("CGPA", 0.0, 10.0, 7.5)
    with col3:
        entrance_exam_score = st.number_input("Entrance Exam Score", 0.0, 100.0, 70.0)
        backlogs = st.number_input("Active/Past Backlogs", 0, 10, 0)

    st.subheader("Technical Proficiency")
    col2_1, col2_2 = st.columns(2)
    with col2_1:
        technical_skill_score = st.number_input("Technical Skill Score (0-100)", 0.0, 100.0, 70.0)
    with col2_2:
        soft_skill_score = st.number_input("Soft Skill Score (0-100)", 0.0, 100.0, 70.0)

    st.markdown("<br>", unsafe_allow_html=True)
    submit_button = st.form_submit_button("Run Prediction", use_container_width=True)

# Bounds & Constants
TRAIN_BOUNDS = {
    'ssc_percentage': (50.0, 95.0),'hsc_percentage': (50.0, 94.0),'degree_percentage': (55.0, 89.0),'cgpa': (5.5, 9.8), 
    'entrance_exam_score': (40.0, 99.0), 'technical_skill_score': (40.0, 99.0),'soft_skill_score': (40.0, 99.0), 'internship_count': (0, 4), 
    'live_projects': (0, 5),'work_experience_months': (0, 24), 'certifications': (0, 5), 'attendance_percentage': (60.0, 99.0),'backlogs': (0, 5)
}

COLUMNS_ORDER = [
    'gender', 'ssc_percentage', 'hsc_percentage', 'degree_percentage', 'cgpa','entrance_exam_score', 'technical_skill_score', 'soft_skill_score',
    'internship_count', 'live_projects', 'work_experience_months','certifications', 'attendance_percentage', 'backlogs', 'extracurricular_activities'
]

# Inference Logic
if submit_button:
    features = {
        'gender': gender, 'ssc_percentage': ssc_percentage, 'hsc_percentage': hsc_percentage,'degree_percentage': degree_percentage, 'cgpa': cgpa, 
        'entrance_exam_score': entrance_exam_score,'backlogs': backlogs, 'internship_count': internship_count, 'live_projects': live_projects,
        'work_experience_months': work_experience_months, 'technical_skill_score': technical_skill_score,'soft_skill_score': soft_skill_score, 
        'certifications': certifications,'attendance_percentage': attendance_percentage, 'extracurricular_activities': extracurricular_activities
    }
    
    out_of_bounds_msgs = []
    for col, (min_val, max_val) in TRAIN_BOUNDS.items():
        val = features[col]
        if val < min_val or val > max_val:
            out_of_bounds_msgs.append(f"*{col}*: {val} (Normal Range: {min_val} - {max_val})")
            
    if out_of_bounds_msgs:
        st.warning("*Extrapolation Warning:* Certain inputs exceed the historical training data range.\n\n" + "\n".join([f"- {msg}" for msg in out_of_bounds_msgs]))

    try:
        with st.spinner('Processing candidate data...'):
            df = pd.DataFrame([features], columns=COLUMNS_ORDER)
            df[df.select_dtypes(['int64', 'int32']).columns] = df.select_dtypes(['int64', 'int32']).astype(float)

            placement_pred = int(clf_pipeline.predict(df)[0])
            placement_prob = float(clf_pipeline.predict_proba(df)[0][1])
            placement_status = "Placed" if placement_pred == 1 else "Not Placed"

            salary_pred = 0.0
            if placement_pred == 1:
                reg_df = df.copy()
                for col, (min_val, max_val) in TRAIN_BOUNDS.items():
                    reg_df[col] = reg_df[col].clip(lower=min_val, upper=max_val)
                    salary_pred = max(0.0, float(reg_pipeline.predict(reg_df)[0]))

            st.markdown("---")
            st.subheader("Analysis Results")
            col_chart, col_metric = st.columns(2)
            
            with col_chart:
                fig = go.Figure(go.Pie(values=[placement_prob, 1-placement_prob], labels=['Placed', 'Not Placed'], hole=.7, marker_colors=['#2563eb', '#e2e8f0'], textinfo='none', hoverinfo='label+percent'))
                fig.update_layout(annotations=[dict(text=f'{placement_prob*100:.1f}%', x=0.5, y=0.5, font_size=26, showarrow=False, font_family="Arial Black")], margin=dict(l=0, r=0, t=0, b=0), height=280, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                if placement_pred:
                    st.success(f"Status Target: {placement_status}")
                else:
                    st.error(f"Status Target: {placement_status}")

            with col_metric:
                st.markdown("<br><br>", unsafe_allow_html=True) 
                st.metric("Estimated Salary (LPA)", f"₹ {salary_pred:.2f} Lakhs")
                if placement_pred:
                    st.info("Estimated based on historical placement data with clipping applied.")
                else:
                    st.warning("Salary estimate is unavailable for non-placed projections.")
            
            st.markdown("---")
            st.subheader("Candidate Profile Breakdown")
            cats = ['Tech Skills', 'Soft Skills', 'SSC %','HSC %', 'Degree', 'CGPA', 'Attendance %']
            vals = [technical_skill_score, soft_skill_score, ssc_percentage, hsc_percentage, degree_percentage, cgpa * 10, attendance_percentage]
            
            fig_radar = go.Figure(go.Scatterpolar(r=vals, theta=cats, fill='toself', fillcolor='rgba(37, 99, 235, 0.2)', line_color='#2563eb', name='Candidate Stats'))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, margin=dict(l=40, r=40, t=60, b=40), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_radar, use_container_width=True)
            
    except Exception as e:
        st.error(f"Failed to process prediction. Error: {str(e)}")
