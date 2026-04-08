from pathlib import Path
import json
import pickle
import traceback

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent


st.set_page_config(page_title="Heart Stroke Risk", layout="wide")

st.markdown(
    """
    <style>
      .main .block-container{
        padding-top: 4px;
        padding-bottom: 6px;
        padding-left: 10px;
        padding-right: 10px;
      }
      .stButton>button { padding: .45rem .8rem; }
      .stSlider, .stNumberInput, .stSelectbox { margin-bottom: 0.35rem; }
      h1 { font-size: 20px; margin: 4px 0 6px 0; }
      .result-card { padding:10px; border-radius:8px; margin-top:6px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    with open(BASE_DIR / "classifier.pkl", "rb") as file:
        model = pickle.load(file)
    with open(BASE_DIR / "scaler.pkl", "rb") as file:
        scaler = pickle.load(file)
    with open(BASE_DIR / "features.json", "r") as file:
        features = json.load(file)
    return model, scaler, features


def build_input_row(features_list, inputs):
    row = pd.DataFrame([[0.0] * len(features_list)], columns=features_list)

    for column in ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak", "FastingBS"]:
        if column in row.columns and column in inputs:
            row.at[0, column] = float(inputs[column])

    if "Sex_M" in row.columns and "Sex" in inputs:
        row.at[0, "Sex_M"] = 1.0 if str(inputs["Sex"]).upper().startswith("M") else 0.0
    elif "Sex_F" in row.columns and "Sex" in inputs:
        row.at[0, "Sex_F"] = 1.0 if str(inputs["Sex"]).upper().startswith("F") else 0.0

    chest_pain_columns = [column for column in row.columns if column.startswith("ChestPainType_")]
    if chest_pain_columns and "ChestPainType" in inputs:
        for column in chest_pain_columns:
            label = column.split("_", 1)[1]
            row.at[0, column] = 1.0 if inputs["ChestPainType"] == label else 0.0

    resting_ecg_columns = [column for column in row.columns if column.startswith("RestingECG_")]
    if resting_ecg_columns and "RestingECG" in inputs:
        for column in resting_ecg_columns:
            label = column.split("_", 1)[1]
            row.at[0, column] = 1.0 if inputs["RestingECG"] == label else 0.0
    elif "RestingECG" in row.columns and "RestingECG" in inputs:
        mapping = {"Normal": 0, "ST": 1, "LVH": 2}
        row.at[0, "RestingECG"] = float(mapping.get(inputs["RestingECG"], 0))

    if "ExerciseAngina_Y" in row.columns and "ExerciseAngina" in inputs:
        row.at[0, "ExerciseAngina_Y"] = (
            1.0 if str(inputs["ExerciseAngina"]).upper().startswith("Y") else 0.0
        )

    st_slope_columns = [column for column in row.columns if column.startswith("ST_Slope_")]
    if st_slope_columns and "ST_Slope" in inputs:
        for column in st_slope_columns:
            label = column.split("_", 1)[1]
            row.at[0, column] = 1.0 if inputs["ST_Slope"] == label else 0.0

    return row.fillna(0.0).astype(float)


def result_card_html(prediction, probability):
    probability_text = f"{probability:.3f}" if probability is not None else "N/A"
    if prediction == 1:
        return (
            "<div class='result-card' style='background:#fff0f0;color:#a30;'>"
            f"<b>HIGH RISK (1)</b> &nbsp; Prob: {probability_text}</div>"
        )
    return (
        "<div class='result-card' style='background:#f0fff6;color:#0a7;'>"
        f"<b>LOW RISK (0)</b> &nbsp; Prob: {probability_text}</div>"
    )


load_error = None
try:
    model, scaler, features = load_artifacts()
except Exception:
    model = scaler = features = None
    load_error = traceback.format_exc()


st.markdown("<h1>Heart Stroke Risk Prediction</h1>", unsafe_allow_html=True)

if load_error:
    st.error("Failed to load model files. Keep classifier.pkl, scaler.pkl, and features.json beside app.py.")
    st.code(load_error)
    st.stop()


column1, column2, column3 = st.columns(3)

with column1:
    age = st.number_input("Age", min_value=18, max_value=100, value=40, step=1, key="age")
    sex = st.selectbox("Sex", ["M", "F"], key="sex")
    chest_pain_type = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"], key="cp")

with column2:
    resting_bp = st.number_input("RestingBP (mm Hg)", min_value=0, max_value=300, value=120, key="rbp")
    cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=0, max_value=1500, value=200, key="chol")
    fasting_bs = st.selectbox("FastingBS (>120 mg/dL)", [0, 1], key="fbs")

with column3:
    resting_ecg = st.selectbox("RestingECG", ["Normal", "ST", "LVH"], key="recg")
    max_hr = st.number_input("MaxHR", min_value=30, max_value=250, value=150, key="maxhr")
    exercise_angina = st.selectbox("ExerciseAngina", ["Y", "N"], key="ea")


row1, row2, row3 = st.columns([2, 1, 1])
with row1:
    oldpeak = st.number_input(
        "Oldpeak (ST Depression)",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        format="%.2f",
        key="old",
    )
with row2:
    st_slope = st.selectbox("ST_Slope", ["Up", "Flat", "Down"], key="slope")
with row3:
    threshold = st.slider(
        "Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        key="threshold",
        help="Probability threshold used to mark High Risk.",
    )


inputs = {
    "Age": age,
    "Sex": sex,
    "ChestPainType": chest_pain_type,
    "RestingBP": resting_bp,
    "Cholesterol": cholesterol,
    "FastingBS": fasting_bs,
    "RestingECG": resting_ecg,
    "MaxHR": max_hr,
    "ExerciseAngina": exercise_angina,
    "Oldpeak": oldpeak,
    "ST_Slope": st_slope,
}


button_left, button_center, button_right = st.columns(3)
with button_center:
    predict_clicked = st.button("Predict", use_container_width=True)


if predict_clicked:
    try:
        input_row = build_input_row(features, inputs)
        non_zero_features = input_row.loc[:, (input_row != 0).any(axis=0)].T

        with st.expander("Input features (non-zero)"):
            st.write(non_zero_features)

        scaled_row = scaler.transform(input_row)

        probability = None
        if hasattr(model, "predict_proba"):
            try:
                probabilities = model.predict_proba(scaled_row)[0]
                probability = probabilities[1] if len(probabilities) > 1 else probabilities[0]
            except Exception:
                probability = None

        if probability is not None:
            prediction = 1 if probability >= threshold else 0
        else:
            prediction = int(model.predict(scaled_row)[0])

        st.markdown(result_card_html(prediction, probability), unsafe_allow_html=True)

    except Exception:
        st.error("Prediction failed. Details are below.")
        st.code(traceback.format_exc())


st.markdown(
    "<div style='text-align:center; color:gray; font-size:12px; margin-top:6px;'>Developed by Mohammad Rashid</div>",
    unsafe_allow_html=True,
)
