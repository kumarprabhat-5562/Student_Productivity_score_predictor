import joblib
import shap

MODEL_PATH = "models/productivity_model.pkl"


def load_model():
    model = joblib.load(MODEL_PATH)
    return model


def load_explainer(model):
    return shap.TreeExplainer(model)