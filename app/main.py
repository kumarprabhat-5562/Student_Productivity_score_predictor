from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import pandas as pd
import joblib
import numpy as np
import shap
import plotly.express as px
import os

app = FastAPI(title="Student Productivity AI Dashboard")

# Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")


model = joblib.load("models/productivity_model.pkl")
explainer = shap.TreeExplainer(model)


def calculate_confidence_interval(input_df):
    """Calculate 5th and 95th percentile from tree predictions"""
    all_preds = np.array([tree.predict(input_df)[0] for tree in model.estimators_])
    lower = np.percentile(all_preds, 5)
    upper = np.percentile(all_preds, 95)
    return round(float(lower), 2), round(float(upper), 2)


def get_performance_label(score):
    if score >= 75:
        return "High"
    elif score >= 50:
        return "Medium"
    else:
        return "Low"


def generate_shap_plot(input_df):
    shap_values = explainer(input_df)

    shap_df = pd.DataFrame({
        "Feature": input_df.columns,
        "Impact": shap_values.values[0]
    })

    fig = px.bar(
        shap_df,
        x="Impact",
        y="Feature",
        orientation='h',
        title="Feature Contribution (SHAP)",
        color="Impact",
        color_continuous_scale="Blues"
    )

    shap_path = "static/shap_plot.html"
    fig.write_html(shap_path, include_plotlyjs="cdn")

    return shap_path


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "prediction": None,
        "performance": "",
        "lower": None,
        "upper": None,
        "positive_impacts": [],
        "negative_impacts": [],
        "suggestions": [],
        "badge_class": ""
    })


@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    study_hours_per_day: float = Form(...),
    sleep_hours: float = Form(...),
    phone_usage_hours: float = Form(...),
    total_distraction_hours: float = Form(...),
    attendance_percentage: float = Form(...)
):

    # Compute derived features internally
    study_efficiency = study_hours_per_day / (
        study_hours_per_day + total_distraction_hours + 1e-5
    )

    balance_score = (
        (sleep_hours * 0.4) +
        (attendance_percentage * 0.3 / 100) +
        (study_efficiency * 0.3)
    )

    # Create DataFrame in correct feature order
    caffeine_sleep_ratio = 0  # Placeholder for future feature
    input_df = pd.DataFrame([{
        "study_hours_per_day": study_hours_per_day,
        "sleep_hours": sleep_hours,
        "balance_score": balance_score,
        "study_efficiency": study_efficiency,
        "phone_usage_hours": phone_usage_hours,
        "total_distraction_hours": total_distraction_hours,
        "attendance_percentage": attendance_percentage,
        "caffeine_sleep_ratio": caffeine_sleep_ratio
    }])
    input_df = input_df[model.feature_names_in_]

    # Prediction
    prediction = model.predict(input_df)[0]
    prediction = round(float(prediction), 2)

    # Confidence Interval
    lower, upper = calculate_confidence_interval(input_df)

    # Performance Label
    performance = get_performance_label(prediction)

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "prediction": prediction,
        "lower": lower,
        "upper": upper,
        "performance": performance,
        "positive_impacts": [],
        "negative_impacts": [],
        "suggestions": [],
        "badge_class": ""
    })