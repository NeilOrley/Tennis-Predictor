import pandas as pd
import joblib
import os
from datetime import datetime
from features import enrich_features, load_h2h_dict, load_recent_form_dict


def predict_today_matches(model_path="models/tennis_win_predictor.pkl",
                          matches_dir="data",
                          output_dir="predictions"):
    """
    Charge les matchs du jour, applique le modèle de prédiction et sauvegarde les résultats.
    """
    today_str = datetime.today().strftime("%Y-%m-%d")
    match_file = os.path.join(matches_dir, f"matches_{today_str}.csv")

    if not os.path.exists(match_file):
        raise FileNotFoundError(f"Aucun fichier trouvé pour aujourd'hui : {match_file}")

    df = pd.read_csv(match_file)
    h2h_dict = load_h2h_dict()
    form_dict = load_recent_form_dict()

    df = enrich_features(df, h2h_dict=h2h_dict, recent_form_dict=form_dict)

    model = joblib.load(model_path)
    proba = model.predict_proba(df)[:, 1]  # Probabilité que Player_1 gagne

    df_result = df.copy()
    df_result["Proba_Player1"] = proba
    df_result["Predicted_Winner"] = df_result.apply(
        lambda row: row["Player_1"] if row["Proba_Player1"] >= 0.5 else row["Player_2"], axis=1
    )

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"predictions_{today_str}.csv")
    df_result.to_csv(output_path, index=False)
    print(f"✅ Prédictions sauvegardées dans : {output_path}")

    return df_result[["Player_1", "Player_2", "Proba_Player1", "Predicted_Winner"]]
