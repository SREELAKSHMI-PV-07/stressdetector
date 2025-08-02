import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.title("🧠 Stress & Mood Stability Detector")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Stress Level Detection Based on Daily Activities.csv")
    df["Social Support"].fillna(df["Social Support"].mode()[0], inplace=True)
    return df

df = load_data()
st.write("### Dataset Preview", df.head())

# Preprocessing
df_encoded = pd.get_dummies(df)
X = df_encoded.drop(['Mood Stability_Stable', 'Mood Stability_Unstable'], axis=1)
y = df_encoded['Mood Stability_Stable']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
st.write("### Model Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
st.write("### Classification Report")
st.text(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
st.write("### Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=["Unstable", "Stable"], yticklabels=["Unstable", "Stable"])
st.pyplot(fig)

# User Input
st.sidebar.header("📋 Enter Your Details")
fields = ['Age', 'Gender', 'Work hours', 'Screen time', 'Sleep time', 'Exercise frequency',
          'Mood Stability', 'Fatigue level', 'Headache', 'Work_life Balance', 'Social Support']
user_data = {field: st.sidebar.text_input(field) for field in fields}

if st.sidebar.button("Predict"):
    input_df = pd.DataFrame([user_data])
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=X_train.columns, fill_value=0)
    prediction = model.predict(input_encoded)[0]
    mood_status = "Stable" if prediction == 1 else "Unstable"
    st.success(f"🧠 Predicted Mood Stability: **{mood_status}**")