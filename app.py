from pathlib import Path
import json
import pickle

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_artifacts():
    with open(BASE_DIR / "classifier.pkl", "rb") as file:
        model = pickle.load(file)

    with open(BASE_DIR / "scaler.pkl", "rb") as file:
        scaler = pickle.load(file)

    with open(BASE_DIR / "features.json", "r") as file:
        features = json.load(file)

    return model, scaler, features


def build_input_row(
    features_list,
    age,
    sex,
    chest_pain,
    resting_bp,
    cholesterol,
    fasting_bs,
    resting_ecg,
    max_hr,
    oldpeak,
    exercise_angina,
    st_slope,
):
    input_df = pd.DataFrame([[0.0] * len(features_list)], columns=features_list)

    numeric_values = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
    }

    for column, value in numeric_values.items():
        if column in input_df.columns:
            input_df.at[0, column] = float(value)

    if "Sex_M" in input_df.columns:
        input_df.at[0, "Sex_M"] = 1.0 if sex == "M" else 0.0

    chest_pain_column = f"ChestPainType_{chest_pain}"
    if chest_pain_column in input_df.columns:
        input_df.at[0, chest_pain_column] = 1.0

    resting_ecg_column = f"RestingECG_{resting_ecg}"
    if resting_ecg_column in input_df.columns:
        input_df.at[0, resting_ecg_column] = 1.0

    if "ExerciseAngina_Y" in input_df.columns:
        input_df.at[0, "ExerciseAngina_Y"] = 1.0 if exercise_angina == "Y" else 0.0

    st_slope_column = f"ST_Slope_{st_slope}"
    if st_slope_column in input_df.columns:
        input_df.at[0, st_slope_column] = 1.0

    return input_df.astype(float)


def predict_heart_disease(
    model,
    scaler,
    features,
    age,
    sex,
    chest_pain,
    resting_bp,
    cholesterol,
    fasting_bs,
    resting_ecg,
    max_hr,
    oldpeak,
    exercise_angina,
    st_slope,
):
    input_df = build_input_row(
        features,
        age,
        sex,
        chest_pain,
        resting_bp,
        cholesterol,
        fasting_bs,
        resting_ecg,
        max_hr,
        oldpeak,
        exercise_angina,
        st_slope,
    )
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    return int(prediction[0])


def main():
    st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

    model, scaler, features = load_artifacts()

    st.title("Heart Disease Prediction App")
    st.markdown(
        """
        <div style="background-color:#0a4b9b;padding:10px;border-radius:6px;">
            <h3 style="color:white;margin:4px;">Enter patient details for prediction</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    age = st.slider("Age", 18, 100, 40)
    sex = st.selectbox("Sex", ["M", "F"])
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
    resting_bp = st.number_input("RestingBP (mm Hg)", min_value=0, max_value=300, value=120)
    cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=0, max_value=1000, value=200)
    fasting_bs = st.selectbox("FastingBS > 120 mg/dL", [0, 1])
    resting_ecg = st.selectbox("RestingECG", ["Normal", "ST", "LVH"])
    max_hr = st.slider("MaxHR", 60, 220, 150)
    oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
    exercise_angina = st.selectbox("ExerciseAngina", ["Y", "N"])
    st_slope = st.selectbox("ST_Slope", ["Up", "Flat", "Down"])

    if st.button("Predict"):
        result = predict_heart_disease(
            model,
            scaler,
            features,
            age,
            sex,
            chest_pain,
            resting_bp,
            cholesterol,
            fasting_bs,
            resting_ecg,
            max_hr,
            oldpeak,
            exercise_angina,
            st_slope,
        )

        if result == 1:
            st.error("Prediction: Heart Disease")
        else:
            st.success("Prediction: No Heart Disease")

    st.markdown("---")
    st.caption(f"Loaded model with {len(features)} input features.")


if __name__ == "__main__":
    main()
