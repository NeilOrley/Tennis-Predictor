import pandas as pd
import argparse
import joblib
import os
from datetime import datetime
from features import enrich_features
from fetch_matches_today import fetch_flashscore_matches, fetch_flashscore_rankings


def load_model(model_path="models/tennis_win_predictor.pkl"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modèle introuvable : {model_path}")
    return joblib.load(model_path)


def predict_all_matches(model, df_matches):
    # Enrichir les données avec les features (sans utiliser les cotes)
    h2h_dict = joblib.load("models/h2h_dict.pkl")
    form_dict = joblib.load("models/recent_form_dict.pkl")

    df_enriched = enrich_features(df_matches.copy(), h2h_dict, form_dict)

    # Prédiction des probabilités (proba que Player_1 gagne)
    proba_p1 = model.predict_proba(df_enriched)[:, 1]
    df_matches["Proba_Player1"] = proba_p1
    df_matches["Predicted_Winner"] = df_matches.apply(
        lambda row: row["Player_1"] if row["Proba_Player1"] >= 0.5 else row["Player_2"], axis=1
    )

    return df_matches


def main():
    parser = argparse.ArgumentParser(description="Prédire tous les matchs du jour (sans odds)")
    parser.add_argument("--model", default="models/tennis_win_predictor.pkl", help="Chemin vers le modèle .pkl")
    parser.add_argument("--output_dir", default="predictions", help="Répertoire de sortie des prédictions")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Chargement des données (matchs + classement)
    matches_df = fetch_flashscore_matches()
    _ = fetch_flashscore_rankings()  # éventuellement utile plus tard

    # Chargement du modèle
    model = load_model(args.model)

    # Prédiction
    predictions = predict_all_matches(model, matches_df)

    today_str = datetime.today().strftime("%Y-%m-%d")
    out_path = os.path.join(args.output_dir, f"predictions_{today_str}.csv")
    predictions.to_csv(out_path, index=False)
    print(f"✅ Prédictions sauvegardées dans : {out_path}")


if __name__ == "__main__":
    main()
