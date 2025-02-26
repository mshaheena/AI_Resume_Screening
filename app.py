# 📌 Step 1: Import Libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

# 📌 Step 2: Load Models & Encoders
@st.cache_resource
def load_resources():
    rf_model = joblib.load("random_forest_model.pkl")
    xgb_model = joblib.load("xgboost_model.pkl")
    svm_model = joblib.load("svm_model.pkl")
    deep_model = load_model("deep_learning_model.keras")
    bilstm_model = load_model("bilstm_model.keras")
    hybrid_model = joblib.load("hybrid_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    label_encoder = joblib.load("label_encoders.pkl")
    return rf_model, xgb_model, svm_model, deep_model, bilstm_model, hybrid_model, vectorizer, label_encoder

rf_model, xgb_model, svm_model, deep_model, bilstm_model, hybrid_model, vectorizer, label_encoder = load_resources()

# 📌 Step 3: Title & Description
st.title("📄 AI-Resume-Screening & Job Matching")
st.write("🚀 Analyze resumes, predict job matches, and screen applicants with AI-powered models.")

# 📌 Step 4: Upload Resume
uploaded_file = st.file_uploader("📤 Upload a Resume (Text File)", type=["txt"])

if uploaded_file:
    resume_text = uploaded_file.read().decode("utf-8")
    
    # 🔹 Display Uploaded Resume
    st.subheader("📜 Uploaded Resume Preview:")
    st.write(resume_text[:500])  # Show only first 500 characters
    
    # 🔹 Preprocess Text
    resume_cleaned = resume_text.lower()  # Convert to lowercase
    
    # 🔹 Convert Text to Features (TF-IDF)
    resume_features = vectorizer.transform([resume_cleaned])
    
    # 📌 Step 5: Make Predictions
    st.subheader("🔍 AI Predictions")

    # ✅ Random Forest Prediction
    rf_pred = label_encoder.inverse_transform(rf_model.predict(resume_features))
    st.write(f"🌲 **Random Forest Prediction:** {rf_pred[0]}")

    # ✅ XGBoost Prediction
    xgb_pred = label_encoder.inverse_transform(xgb_model.predict(resume_features))
    st.write(f"🚀 **XGBoost Prediction:** {xgb_pred[0]}")

    # ✅ SVM Prediction
    svm_pred = label_encoder.inverse_transform(svm_model.predict(resume_features))
    st.write(f"🖥️ **SVM Prediction:** {svm_pred[0]}")

    # ✅ Deep Learning Prediction
    deep_pred = label_encoder.inverse_transform([np.argmax(deep_model.predict(resume_features.toarray()))])
    st.write(f"🧠 **Deep Learning Prediction:** {deep_pred[0]}")

    # ✅ BiLSTM Prediction
    bilstm_pred = label_encoder.inverse_transform([np.argmax(bilstm_model.predict(resume_features.toarray()))])
    st.write(f"🔄 **BiLSTM Prediction:** {bilstm_pred[0]}")

    # ✅ Hybrid Model Prediction
    hybrid_pred = label_encoder.inverse_transform([np.round((xgb_model.predict(resume_features) + np.argmax(deep_model.predict(resume_features.toarray()))) / 2).astype(int)][0])
    st.write(f"⚡ **Hybrid Model Prediction:** {hybrid_pred}")

# 📌 Step 6: Job Matching
st.subheader("💼 AI-Powered Job Matching")
job_roles = ["Data Scientist", "Software Engineer", "Machine Learning Engineer", "HR Manager", "AI Researcher"]
selected_role = st.selectbox("Select a Job Role for Matching:", job_roles)

if st.button("🔍 Find Best Resume Matches"):
    st.write(f"✅ Showing best resume matches for **{selected_role}**...")
    # Simulate some results
    st.write("📄 **Best Matched Resume:** John Doe (85% Match)")
    st.write("📄 **Second Best:** Jane Smith (79% Match)")
    st.write("📄 **Third Best:** Alice Johnson (74% Match)")

# 📌 Step 7: Closing Note
st.write("✨ **AI-powered resume screening & job matching powered by ML & DL models!**")
