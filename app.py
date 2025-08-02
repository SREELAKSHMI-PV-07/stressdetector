
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load model and dataset
model = joblib.load("stress_model.pkl")
df = pd.read_csv("cleaned_stress_dataset.csv")

# Encode target
le = LabelEncoder()
df['mood_stability'] = le.fit_transform(df['mood_stability'])

# Separate features and target
X = df.drop('mood_stability', axis=1)
y = df['mood_stability']

# User Interface
st.title("Stress & Mood Stability Predictor")

menu = st.sidebar.radio("Choose Mode", ["🎯 Predict Mood", "📊 Model Evaluation"])

if menu == "🎯 Predict Mood":
    st.subheader("Enter Your Details")

    # Collect user inputs (adjust according to your dataset)
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 15, 80, 25)
    sleep_time = st.slider("Sleep Time (hours)", 0, 12, 7)
    screen_time = st.slider("Screen Time (hours)", 0, 15, 6)
    work_hours = st.slider("Work/Study Hours", 0, 16, 8)
    exercise = st.selectbox("Exercise Frequency", ["None", "Rarely", "Regularly"])
    diet = st.selectbox("Diet Quality", ["Poor", "Average", "Good"])

    # Encode user inputs like in training
    gender = 1 if gender == "Male" else 0
    exercise_map = {"None": 0, "Rarely": 1, "Regularly": 2}
    diet_map = {"Poor": 0, "Average": 1, "Good": 2}

    user_data = [[gender, age, sleep_time, screen_time, work_hours, exercise_map[exercise], diet_map[diet]]]

    if st.button("Predict"):
        prediction = model.predict(user_data)
        label = le.inverse_transform(prediction)[0]
        st.success(f"✅ Predicted Mood Stability: **{label}**")

elif menu == "📊 Model Evaluation":
    st.subheader("Model Evaluation on Dataset")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Classification report
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred, target_names=le.classes_))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    fig, ax = plt.subplots()
    disp.plot(ax=ax, cmap='Blues')
    st.pyplot(fig)
