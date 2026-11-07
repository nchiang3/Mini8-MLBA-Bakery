# Ice Cream Sales Predictor - Week 3

This Streamlit app predicts daily ice cream sales using a linear regression model trained on synthetic data.

## Predictors
- **Temperature:** Forecasted daily high temperature (°F).
- **High Demand Days:** Friday, Saturday, and Sunday are classified as high-demand days (1), while Monday through Thursday are regular days (0).

## How It Works
1. The user selects a city and date.
2. The app retrieves the forecasted temperature using the Open-Meteo API.
3. The app determines whether the selected date is a high-demand day.
4. The saved regression model predicts the number of ice cream cones sold.
5. The app displays the prediction and a business recommendation.

## Model
The model was trained using synthetic ice cream sales data. High-demand days were simulated to have approximately 35 additional sales compared to regular days.

## Files
- `app.py` — Streamlit web application
- `model.pkl` — Trained linear regression model
- `requirements.txt` — Python dependencies
