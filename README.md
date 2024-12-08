# Rain Prediction Model

A machine learning model built to predict rainfall using various weather parameters.

## Overview

This project uses a **Random Forest Classifier** to predict whether it will rain or not based on various meteorological features. The model is trained on past weather data and can make  predictions on behalf of it weatherit will rain/no rain.

## Features Used

The model uses the following weather parameters for prediction:
- Pressure
- Dewpoint
- Humidity
- Cloud Coverage
- Sunshine
- Wind Speed
- Wind Direction

## Data Processing Steps

1. Data Loading and Cleaning
   - Loads data from 'Rainfall.csv'
   - Removes extra spaces in column names
   - Handles missing values in wind direction and wind speed
   - Converts rainfall labels to binary (1 for yes, 0 for no)

2. Data Visualisation
   - Removes highly correlated features (maxtemp, mintemp, temperature)
   - Performs downsampling to handle class imbalance

3. Model Training
   - Uses RandomForestClassifier with GridSearchCV for hyperparameter tuning
   - Performs cross-validation to ensure model reliability
   - Evaluates model using confusion matrix and classification report

## Model Performance

The model is evaluated using:
- Cross-validation scores
- Test set accuracy
- Confusion matrix
- Classification report
# Load the Modal:
 with open("model.pkl", "rb") as file:
 model_data = pickle.load(file)
 model = model_data["model"]
 features_names = model_data["features"]

## Model Storage

The trained model is saved as 'model.pkl' using pickle, which includes:
- The trained RandomForestClassifier model
- Feature names used for prediction

## Example Input Data
input_data = [1015.9, 19.9, 95, 81, 0, 40, 13.7]
input_df = pd.DataFrame([input_data], columns=features_names)
