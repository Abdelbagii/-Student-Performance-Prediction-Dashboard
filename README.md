AI-Based Student Academic Performance Prediction Dashboard

Project Overview

The **AI-Based Student Academic Performance Prediction Dashboard** is a machine learning web application that predicts student academic performance and helps identify students who may be at risk of low performance.

This project uses student academic and personal data, such as study time, absences, family support, school support, and previous grades, to classify students into performance levels:
- **High**
- **Medium**
- **Low**

The system also provides an interactive dashboard for exploring the dataset, visualizing trends, and monitoring at-risk students.

## Features
- Student performance prediction using machine learning
- Interactive dashboard built with Streamlit
- Dataset overview and exploration
- Data analysis visualizations
- At-risk student identification
- Easy-to-use prediction form



## Tools Used
- **Python**
- **Streamlit**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Plotly**
- **Joblib**

## Project Structure

student-performance-dashboard/
│
├── data/
│   └── student-mat.csv
│
├── model/
│   ├── student_performance_model.pkl
│   ├── label_encoders.pkl
│   └── target_encoder.pkl
│
├── assets/
├── app/
│   └── streamlit_app.py
│
├── train_model.py
├── requirements.txt
└── README.md
