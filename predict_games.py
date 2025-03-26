import joblib
import pandas as pd

def predict_total_games_class(input_dict, model_path="models/tennis_total_games_predictor.pkl"):
    model = joblib.load(model_path)
    X_input = pd.DataFrame([input_dict])
    prediction = model.predict(X_input)[0]
    return prediction