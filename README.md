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

- `0` = Low risk / No heart disease predicted
- `1` = High risk / Heart disease predicted

The app can also show a probability score when the model supports it.

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
