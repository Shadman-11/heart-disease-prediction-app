import streamlit as st
import pandas as pd
import joblib

# Load saved files
model = joblib.load("Naive_Bayes_heart.pkl")
expected_columns = joblib.load("columns.pkl")

st.title("Heart Disease Prediction")
st.markdown("Enter patient information to predict heart disease risk.")

age = st.slider("Age", 20, 100, 50)

sex = st.selectbox(
    "Sex",
    [1, 0],
    format_func=lambda x: "Male" if x == 1 else "Female"
)

chest_pain_type = st.selectbox(
    "Chest Pain Type",
    [1, 2, 3, 4],
    format_func=lambda x: {
        1: "Type 1",
        2: "Type 2",
        3: "Type 3",
        4: "Type 4"
    }[x]
)

bp = st.number_input("BP", min_value=80, max_value=220, value=120)

cholesterol = st.number_input(
    "Cholesterol",
    min_value=100,
    max_value=700,
    value=200
)

fbs_over_120 = st.selectbox(
    "FBS over 120",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

ekg_results = st.selectbox(
    "EKG Results",
    [0, 1, 2],
    format_func=lambda x: f"Result {x}"
)

max_hr = st.slider("Max HR", 60, 220, 150)

exercise_angina = st.selectbox(
    "Exercise Angina",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

st_depression = st.number_input(
    "ST Depression",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.1
)

slope_of_st = st.selectbox(
    "Slope of ST",
    [1, 2, 3],
    format_func=lambda x: f"Slope {x}"
)

number_of_vessels_fluro = st.selectbox(
    "Number of Vessels Fluro",
    [0, 1, 2, 3]
)

thallium = st.selectbox(
    "Thallium",
    [3, 6, 7],
    format_func=lambda x: f"Thallium {x}"
)

if st.button("Predict"):

    input_data = {
        "Age": age,
        "Sex": sex,
        "Chest pain type": chest_pain_type,
        "BP": bp,
        "Cholesterol": cholesterol,
        "FBS over 120": fbs_over_120,
        "EKG results": ekg_results,
        "Max HR": max_hr,
        "Exercise angina": exercise_angina,
        "ST depression": st_depression,
        "Slope of ST": slope_of_st,
        "Number of vessels fluro": number_of_vessels_fluro,
        "Thallium": thallium
    }

    input_df = pd.DataFrame([input_data])

    # Correct column order
    input_df = input_df[expected_columns]

    # No scaling here, because this model was trained on raw data
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    high_risk_probability = probability[1]

    st.write("High Risk Probability:", round(high_risk_probability * 100, 2), "%")

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")