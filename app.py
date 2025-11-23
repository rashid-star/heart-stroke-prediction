import streamlit as st
import pandas as pd
import pickle, json, traceback

# --- Page config and compact CSS ---
st.set_page_config(page_title="Heart Stroke Risk", layout="wide")

# Compact styling: remove large top margin/padding and tighten widget spacing
st.markdown(
    """
    <style>
      /* container padding (reduce top/bottom/side spacing) */
      .main .block-container{
        padding-top: 4px;
        padding-bottom: 6px;
        padding-left: 10px;
        padding-right: 10px;
      }
      /* Reduce space between widgets */
      .stButton>button { padding: .45rem .8rem; }
      .stSlider, .stNumberInput, .stSelectbox { margin-bottom: 0.35rem; }
      /* Smaller title size so it doesn't take vertical space */
      h1 { font-size: 20px; margin: 4px 0 6px 0; }
      /* compact result card font */
      .result-card { padding:10px; border-radius:8px; margin-top:6px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Helpers ----------------
@st.cache_data
def load_artifacts():
    """Load model, scaler and features list saved from training (pickle/json)."""
    with open("classifier.pkl","rb") as f:
        clf = pickle.load(f)
    with open("scaler.pkl","rb") as f:
        scaler = pickle.load(f)
    with open("features.json","r") as f:
        features = json.load(f)
    return clf, scaler, features

def build_input_row(features_list, inputs):
    """
    Create a single-row DataFrame with columns ordered as features_list.
    It sets numeric features directly and one-hot columns when present.
    """
    row = pd.DataFrame(columns=features_list)
    row.loc[0] = 0.0

    # Direct numeric features if present
    for col in ['Age','RestingBP','Cholesterol','MaxHR','Oldpeak','FastingBS']:
        if col in row.columns and col in inputs:
            row.at[0, col] = float(inputs[col])

    # Sex (one-hot with drop_first often creates Sex_M)
    if 'Sex_M' in row.columns and 'Sex' in inputs:
        row.at[0,'Sex_M'] = 1.0 if str(inputs['Sex']).upper().startswith('M') else 0.0
    elif 'Sex_F' in row.columns and 'Sex' in inputs:
        row.at[0,'Sex_F'] = 1.0 if str(inputs['Sex']).upper().startswith('F') else 0.0

    # ChestPainType one-hot (set whichever one-hot columns exist)
    cp_cols = [c for c in row.columns if c.startswith('ChestPainType_')]
    if cp_cols and 'ChestPainType' in inputs:
        for c in cp_cols:
            label = c.split('_',1)[1]
            row.at[0,c] = 1.0 if inputs['ChestPainType'] == label else 0.0

    # RestingECG one-hot (or single numeric)
    recg_cols = [c for c in row.columns if c.startswith('RestingECG_')]
    if recg_cols and 'RestingECG' in inputs:
        for c in recg_cols:
            label = c.split('_',1)[1]
            row.at[0,c] = 1.0 if inputs['RestingECG'] == label else 0.0
    if 'RestingECG' in row.columns and 'RestingECG' in inputs and not recg_cols:
        mapping = {'Normal':0, 'ST':1, 'LVH':2}
        row.at[0,'RestingECG'] = float(mapping.get(inputs['RestingECG'], 0))

    # ExerciseAngina (one-hot example)
    if 'ExerciseAngina_Y' in row.columns and 'ExerciseAngina' in inputs:
        row.at[0,'ExerciseAngina_Y'] = 1.0 if str(inputs['ExerciseAngina']).upper().startswith('Y') else 0.0

    # ST_Slope one-hot
    slope_cols = [c for c in row.columns if c.startswith('ST_Slope_')]
    if slope_cols and 'ST_Slope' in inputs:
        for c in slope_cols:
            label = c.split('_',1)[1]
            row.at[0,c] = 1.0 if inputs['ST_Slope'] == label else 0.0

    return row.fillna(0.0).astype(float)

def result_card_html(pred, prob):
    """Return small HTML for result (compact)."""
    prob_text = f"{prob:.3f}" if prob is not None else "N/A"
    if pred == 1:
        return f"<div class='result-card' style='background:#fff0f0;color:#a30;'><b>⚠ HIGH RISK (1)</b> &nbsp; Prob: {prob_text}</div>"
    else:
        return f"<div class='result-card' style='background:#f0fff6;color:#0a7;'><b>✔ LOW RISK (0)</b> &nbsp; Prob: {prob_text}</div>"

# ---------------- Load model artifacts ----------------
load_error = None
try:
    clf, scaler, features = load_artifacts()
except Exception as e:
    clf = scaler = features = None
    load_error = traceback.format_exc()

# --- Top: small title (compact) ---
st.markdown("<h1>❤️ Heart Stroke Risk Prediction </h1>", unsafe_allow_html=True)

if load_error:
    st.error("Failed to load model/scaler/features. Put classifier.pkl, scaler.pkl, features.json here.")
    st.code(load_error)
    st.stop()

# ---------------- Inputs: three equal columns ----------------
# We use 3 columns so inputs are evenly spaced and compact
c1, c2, c3 = st.columns(3)

with c1:
    Age = st.number_input("Age", min_value=18, max_value=100, value=40, step=1, key="age")
    Sex = st.selectbox("Sex", ["M","F"], key="sex")
    ChestPainType = st.selectbox("Chest Pain Type", ["ATA","NAP","TA","ASY"], key="cp")
with c2:
    RestingBP = st.number_input("RestingBP (mm Hg)", min_value=0, max_value=300, value=120, key="rbp")
    Cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=0, max_value=1500, value=200, key="chol")
    FastingBS = st.selectbox("FastingBS (>120 mg/dL)", [0,1], key="fbs")
with c3:
    RestingECG = st.selectbox("RestingECG", ["Normal","ST","LVH"], key="recg")
    MaxHR = st.number_input("MaxHR", min_value=30, max_value=250, value=150, key="maxhr")
    ExerciseAngina = st.selectbox("ExerciseAngina", ["Y","N"], key="ea")

# Single-row input for Oldpeak and ST_Slope placed beneath columns in a single line
r1, r2, r3 = st.columns([2,1,1])  # slightly wider first column for Oldpeak
with r1:
    Oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0, value=1.0, format="%.2f", key="old")
with r2:
    ST_Slope = st.selectbox("ST_Slope", ["Up","Flat","Down"], key="slope")
with r3:
    # threshold slider compact placed small
    threshold = st.slider("Threshold", 0.0, 1.0, 0.5, key="th", help="Probability threshold to mark High Risk")

# Pack inputs into a dict (order doesn't matter here - features.json defines order)
inputs = {
    "Age": Age, "Sex": Sex, "ChestPainType": ChestPainType,
    "RestingBP": RestingBP, "Cholesterol": Cholesterol, "FastingBS": FastingBS,
    "RestingECG": RestingECG, "MaxHR": MaxHR, "ExerciseAngina": ExerciseAngina,
    "Oldpeak": Oldpeak, "ST_Slope": ST_Slope
}

# ------- Predict button centered below inputs -------
# We create 3 columns to center the button (button in middle column)
b1, b2, b3 = st.columns([1,1,1])
with b2:
    pred_clicked = st.button("Predict", key="predict", use_container_width=True)

# Output area: show result below the button
if pred_clicked:
    try:
        # Build the model input row with exact columns/order as training features
        row = build_input_row(features, inputs)

        # Optional: show non-zero features for debugging
        nz = row.loc[:, (row != 0).any(axis=0)].T
        if not nz.empty:
            st.write("Input features (non-zero):")
            st.write(nz)

        # Scale and predict
        scaled = scaler.transform(row)

        prob = None
        # try predict_proba safely
        if hasattr(clf, "predict_proba"):
            try:
                probs = clf.predict_proba(scaled)[0]
                prob = probs[1] if len(probs) > 1 else probs[0]
            except Exception:
                prob = None

        # final class decision: if prob available use threshold else use predict()
        if prob is not None:
            pred = 1 if prob >= threshold else 0
        else:
            pred = int(clf.predict(scaled)[0])

        # show compact result card
        st.markdown(result_card_html(pred, prob), unsafe_allow_html=True)

    except Exception as ex:
        st.error("Prediction failed — see details below.")
        st.code(traceback.format_exc())

# Small footer (very compact)
st.markdown("<div style='text-align:center; color:gray; font-size:12px; margin-top:6px;'> Developed by Mohammad Rashid </div>", unsafe_allow_html=True)
