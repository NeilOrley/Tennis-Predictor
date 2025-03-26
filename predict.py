import joblib
import pandas as pd

def predict_match(input_dict, model_path):
    model = joblib.load(model_path)
    X_input = pd.DataFrame([input_dict])
    probas = model.predict_proba(X_input)[0]
    return {
        "Probabilité victoire Player_1": round(probas[1], 4),
        "Probabilité victoire Player_2": round(probas[0], 4)
    }