# Heart Stroke Risk Prediction

A Streamlit machine learning app that predicts heart disease risk from patient health details.

## Features

- Interactive Streamlit user interface
- Input-based heart disease risk prediction
- Saved machine learning model with scaler and feature metadata
- Probability-based result display with adjustable threshold
- Easy to explain for demos, viva, and college presentation

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn

## Project Files

- `app.py` - Streamlit application
- `Heart_Stroke_Prediction.ipynb` - training and experimentation notebook
- `Heart_Stroke_Risk_Prediction.ipynb` - updated notebook version from the GitHub side
- `HeartDesies.csv` - dataset
- `classifier.pkl` - trained model
- `scaler.pkl` - saved scaler
- `features.json` - feature order used during prediction
- `requirements.txt` - dependencies

## How It Works

1. The dataset is loaded and preprocessed.
2. Categorical values are encoded and features are scaled.
3. A KNN classifier is trained in the notebook.
4. The model, scaler, and feature list are saved.
5. The Streamlit app loads those saved files and predicts the result from user input.

## Output

The result card shows a **stroke probability percentage** with a colour-coded risk level and a brief advisory message:

| Probability | Risk Level | Colour |
|-------------|-----------|--------|
| 0 – <20 % | ✅ Very Low Risk | Green |
| 20 – <40 % | 🟢 Low Risk | Light green |
| 40 – <60 % | 🟡 Moderate Risk | Yellow |
| 60 – <80 % | 🟠 High Risk | Orange |
| 80 – 100 % | ⚠️ Very High Risk | Red |

A visual progress bar and a short health advisory are included in the card.  
The **Threshold** slider lets you adjust the cut-off used to assign the final binary label (High / Low).

## Run Locally

```powershell
python -m venv env
.\env\Scripts\activate
pip install -r requirements.txt
python -m streamlit run app.py
```
## Live Demo

https://hearts-stroke-risk-prediction.streamlit.app/

## GitHub Repository

https://github.com/rashid-star/heart-stroke-prediction

## Author

Mohammad Rashid
