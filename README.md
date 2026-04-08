# Heart Stroke Risk Prediction

A Streamlit web app that predicts heart disease risk from patient health details using a trained K-Nearest Neighbors model.

## Project Overview

This project uses a machine learning workflow to:

- load and preprocess heart disease data
- encode categorical features
- scale the inputs with `StandardScaler`
- train a `KNeighborsClassifier`
- save the trained model and feature metadata
- serve predictions through a Streamlit user interface

The app is designed to make the model easy to explain and demonstrate in class.

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit

## Files

- `app.py` - Streamlit application
- `Heart_Stroke_Prediction.ipynb` - notebook used for training and experimentation
- `HeartDesies.csv` - dataset
- `classifier.pkl` - saved trained model
- `scaler.pkl` - saved scaler
- `features.json` - saved feature order for inference
- `requirements.txt` - project dependencies

## How It Works

1. The notebook prepares the dataset and trains the KNN model.
2. The model, scaler, and feature list are saved using `pickle` and `json`.
3. The Streamlit app loads those saved files.
4. User input is converted into the exact feature format expected by the model.
5. The app returns a prediction for heart disease risk.

## Run Locally

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv env
.\env\Scripts\activate
pip install -r requirements.txt
```

Run the Streamlit app:

```powershell
python -m streamlit run app.py
```

## Sample Input Features

The app accepts values such as:

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Max Heart Rate
- Oldpeak
- Exercise Angina
- ST Slope

## Output

The model predicts one of these outcomes:

- `0` = Low risk / No heart disease predicted
- `1` = High risk / Heart disease predicted

## Deployment

This project can be deployed with Streamlit Community Cloud by linking the GitHub repository and selecting:

- main file: `app.py`
- dependency file: `requirements.txt`

## GitHub

Repository: [rashid-star/heart-stroke-prediction](https://github.com/rashid-star/heart-stroke-prediction)

## Author

Mohammad Rashid
