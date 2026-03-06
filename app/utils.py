import numpy as np
import pandas as pd

def compute_derived_features(data):
    """
    Compute internal engineered features.
    """
    study_efficiency = (
        data["study_hours_per_day"] /
        (data["study_hours_per_day"] + data["total_distraction_hours"] + 1e-5)
    )

    balance_score = (
        (data["sleep_hours"] * 0.4) +
        (data["attendance_percentage"] * 0.3 / 100) +
        (study_efficiency * 0.3)
    )

    data["study_efficiency"] = study_efficiency
    data["balance_score"] = balance_score

    return data

def calculate_confidence_interval(model, input_df):
    all_preds = np.array([
        tree.predict(input_df)[0]
        for tree in model.estimators_
    ])

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

def process_shap_values(explainer, input_df):
    shap_values = explainer(input_df)

    feature_names = input_df.columns
    shap_vals = shap_values.values[0]

    positive_impacts = []
    negative_impacts = []
    suggestions = []

    for feature, value in zip(feature_names, shap_vals):

        explanation = generate_feature_explanation(feature, value)

        if value > 0:
            positive_impacts.append(explanation)
        else:
            negative_impacts.append(explanation)
            suggestion = generate_suggestion(feature)
            if suggestion:
                suggestions.append(suggestion)

    return positive_impacts, negative_impacts, suggestions

def generate_feature_explanation(feature, value):

    impact_strength = abs(round(value, 2))

    explanations = {
        "study_hours_per_day":
            f"Studying more hours is improving your productivity (+{impact_strength}).",

        "sleep_hours":
            f"Your sleep pattern is affecting productivity (impact: {impact_strength}).",

        "phone_usage_hours":
            f"Phone usage is reducing your focus (impact: {impact_strength}).",

        "total_distraction_hours":
            f"High distraction time is lowering productivity (impact: {impact_strength}).",

        "attendance_percentage":
            f"Attendance level contributes to discipline and productivity (impact: {impact_strength}).",

        "study_efficiency":
            f"Your study efficiency ratio influences output quality (impact: {impact_strength}).",

        "balance_score":
            f"Life-study balance affects long-term productivity (impact: {impact_strength})."
    }

    return explanations.get(feature, f"{feature} impact: {impact_strength}")

def generate_suggestion(feature):

    suggestions_map = {
        "phone_usage_hours":
            "Reduce daily phone usage to improve concentration.",

        "total_distraction_hours":
            "Minimize distractions during study sessions.",

        "sleep_hours":
            "Maintain 7-8 hours of quality sleep.",

        "study_hours_per_day":
            "Increase focused study hours gradually.",

        "attendance_percentage":
            "Improve attendance consistency.",

        "study_efficiency":
            "Try focused study techniques like Pomodoro.",

        "balance_score":
            "Maintain a healthy balance between rest and study."
    }

    return suggestions_map.get(feature)