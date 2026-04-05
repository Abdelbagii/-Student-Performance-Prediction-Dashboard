import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(page_title="Student Performance Dashboard", layout="wide")

# Load dataset
df = pd.read_csv("data/student-mat.csv", sep=";")

# Load model files
model = joblib.load("model/student_performance_model.pkl")
label_encoders = joblib.load("model/label_encoders.pkl")
target_encoder = joblib.load("model/target_encoder.pkl")

# Create performance label for dashboard visuals
def categorize_performance(grade):
    if grade >= 15:
        return "High"
    elif grade >= 10:
        return "Medium"
    else:
        return "Low"

df["performance"] = df["G3"].apply(categorize_performance)

st.title("AI-Based Student Academic Performance Prediction Dashboard")
st.markdown("Predict student performance and identify at-risk students using machine learning.")

menu = st.sidebar.radio(
    "Navigation",
    ["Overview", "Data Analysis", "Prediction", "At-Risk Students"]
)

if menu == "Overview":
    st.subheader("Dataset Overview")
    st.write("Number of students:", df.shape[0])
    st.write("Number of features:", df.shape[1] - 1)
    st.dataframe(df.head())

    performance_counts = df["performance"].value_counts().reset_index()
    performance_counts.columns = ["Performance", "Count"]

    fig = px.pie(
        performance_counts,
        names="Performance",
        values="Count",
        title="Performance Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

elif menu == "Data Analysis":
    st.subheader("Data Analysis")

    fig1 = px.histogram(df, x="G3", nbins=20, title="Final Grade Distribution")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.scatter(
        df,
        x="studytime",
        y="G3",
        color="performance",
        title="Study Time vs Final Grade"
    )
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.scatter(
        df,
        x="absences",
        y="G3",
        color="performance",
        title="Absences vs Final Grade"
    )
    st.plotly_chart(fig3, use_container_width=True)

elif menu == "Prediction":
    st.subheader("Student Performance Prediction")

    age = st.slider("Age", 15, 22, 17)
    studytime = st.slider("Study Time (1=low, 4=high)", 1, 4, 2)
    failures = st.slider("Past Class Failures", 0, 4, 0)
    absences = st.slider("Absences", 0, 30, 4)
    schoolsup = st.selectbox("School Support", ["yes", "no"])
    famsup = st.selectbox("Family Support", ["yes", "no"])
    higher = st.selectbox("Wants Higher Education", ["yes", "no"])
    internet = st.selectbox("Internet Access", ["yes", "no"])

    input_dict = {
        "school": "GP",
        "sex": "F",
        "age": age,
        "address": "U",
        "famsize": "GT3",
        "Pstatus": "T",
        "Medu": 2,
        "Fedu": 2,
        "Mjob": "other",
        "Fjob": "other",
        "reason": "course",
        "guardian": "mother",
        "traveltime": 1,
        "studytime": studytime,
        "failures": failures,
        "schoolsup": schoolsup,
        "famsup": famsup,
        "paid": "no",
        "activities": "yes",
        "nursery": "yes",
        "higher": higher,
        "internet": internet,
        "romantic": "no",
        "famrel": 4,
        "freetime": 3,
        "goout": 3,
        "Dalc": 1,
        "Walc": 1,
        "health": 3,
        "absences": absences,
        "G1": 10,
        "G2": 10
    }

    input_df = pd.DataFrame([input_dict])

    for column, encoder in label_encoders.items():
        if column in input_df.columns:
            input_df[column] = encoder.transform(input_df[column])

    prediction = model.predict(input_df)[0]
    predicted_label = target_encoder.inverse_transform([prediction])[0]

    if st.button("Predict Performance"):
        st.success(f"Predicted Performance Level: {predicted_label}")

elif menu == "At-Risk Students":
    st.subheader("At-Risk Students")
    at_risk_df = df[df["performance"] == "Low"][["school", "sex", "age", "studytime", "failures", "absences", "G3"]]
    st.dataframe(at_risk_df)

    fig4 = px.histogram(
        at_risk_df,
        x="absences",
        nbins=15,
        title="Absence Distribution of At-Risk Students"
    )
    st.plotly_chart(fig4, use_container_width=True)