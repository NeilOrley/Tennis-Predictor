
def kelly_fraction(p, b):
    return max((p * (b - 1) - (1 - p)) / (b - 1), 0)


# predict_today_matches.py
import pandas as pd
import joblib
import os
from datetime import datetime
from preprocessing import enrich_features, compute_all_features, load_h2h_dict, load_recent_form_dict
from elo_model import EloModel, update_elo_from_matches
from fetch_data import fetch_flashscore_matches

def predict_today_matches(matches_dir="data", output_dir="predictions"):
    today_str = datetime.today().strftime("%Y-%m-%d")
    match_file = os.path.join(matches_dir, f"matches_{today_str}.csv")

    if not os.path.exists(match_file):
        print("📡 Données du jour non trouvées. Tentative de récupération...")
        fetch_flashscore_matches()

    df = pd.read_csv(match_file)

    h2h_dict = load_h2h_dict()
    form_dict = load_recent_form_dict()
    df = enrich_features(df, h2h_dict=h2h_dict, recent_form_dict=form_dict)
    df = compute_all_features(df)

    use_odds = "Odd_1" in df.columns and "Odd_2" in df.columns and (df["Odd_1"] > 0).all() and (df["Odd_2"] > 0).all()
    model_path = "models/model_with_odds.pkl" if use_odds else "models/model_without_odds.pkl"
    print(f"📁 Modèle ML utilisé : {model_path}")
    ml_model = joblib.load(model_path)

    classes = ml_model.classes_
    if list(classes) == [0, 1]:
        proba_ml_player1 = ml_model.predict_proba(df)[:, 1]
    elif list(classes) == [1, 0]:
        proba_ml_player1 = ml_model.predict_proba(df)[:, 0]
    else:
        raise ValueError(f"Ordre de classes inattendu : {classes}")

    df["Proba_ML_Player1"] = proba_ml_player1
    df["Proba_ML_Player2"] = 1 - proba_ml_player1
    df["Predicted_ML_Winner"] = df.apply(
        lambda row: row["Player_1"] if row["Proba_ML_Player1"] >= 0.5 else row["Player_2"], axis=1
    )
    df["Confidence_ML"] = df["Proba_ML_Player1"].apply(lambda p: abs(p - 0.5) * 200)

    elo_model = EloModel(initial_rating=1500, k=32, factor=400)
    historical_file = os.path.join(matches_dir, "atp_tennis.csv")
    if os.path.exists(historical_file):
        df_hist = pd.read_csv(historical_file)
        df_hist = df_hist.dropna(subset=["Player_1", "Player_2", "Winner", "Date"])
        update_elo_from_matches(df_hist, elo_model)
    else:
        print("Fichier historique introuvable, utilisation des ratings initiaux.")

    elo_predictions = []
    for _, row in df.iterrows():
        prob_elo_player1 = elo_model.predict_match(row["Player_1"], row["Player_2"])
        elo_predictions.append(prob_elo_player1)
    df["Proba_Elo_Player1"] = elo_predictions
    df["Proba_Elo_Player2"] = 1 - df["Proba_Elo_Player1"]
    df["Predicted_Elo_Winner"] = df.apply(
        lambda row: row["Player_1"] if row["Proba_Elo_Player1"] >= 0.5 else row["Player_2"], axis=1
    )

    print("\n🎾 Prédictions des vainqueurs :")
    for _, row in df.iterrows():
        print(f"➡️ {row['Player_1']} vs {row['Player_2']} - Tournoi : {row['Tournament']}")
        print(f"   ML : {row['Player_1']} : {row['Proba_ML_Player1']*100:.2f}%, {row['Player_2']} : {row['Proba_ML_Player2']*100:.2f}% => {row['Predicted_ML_Winner']} (confiance : {row['Confidence_ML']:.1f}%)")
        print(f"   Elo: {row['Player_1']} : {row['Proba_Elo_Player1']*100:.2f}%, {row['Player_2']} : {row['Proba_Elo_Player2']*100:.2f}% => {row['Predicted_Elo_Winner']}")
        print()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"predictions_{today_str}.csv")
    
    # 💰 Kelly Criterion + bankroll simulation
    bankroll = 100.0
    stakes = []
    kellys = []
    evs = []
    bets = []
    results = []
    gains = []
    
    for _, row in df.iterrows():
        odds = row["Odd_1"] if row["Predicted_ML_Winner"] == row["Player_1"] else row["Odd_2"]
        proba = row["Proba_ML_Player1"] if row["Predicted_ML_Winner"] == row["Player_1"] else row["Proba_ML_Player2"]
        kelly = kelly_fraction(proba, odds)
        stake = bankroll * kelly
        expected = stake * (proba * (odds - 1) - (1 - proba))
        bets.append(row["Predicted_ML_Winner"])
        kellys.append(kelly)
        stakes.append(stake)
        evs.append(expected)

        if "Winner" in row and row["Predicted_ML_Winner"] == row["Winner"]:
            gain = stake * (odds - 1)
        else:
            gain = -stake
        gains.append(gain)
        bankroll += gain

    df["Kelly_Fraction"] = kellys
    df["Stake"] = stakes
    df["EV"] = evs
    df["Bet_On"] = bets
    df["Gain"] = gains
    df["Bankroll_After"] = pd.Series(gains).cumsum() + 100.0

    df.to_csv(output_path, index=False)
    print(f"✅ Prédictions sauvegardées dans : {output_path}")

    return df[["Tournament", "Player_1", "Player_2", 
               "Proba_ML_Player1", "Proba_ML_Player2", "Predicted_ML_Winner", "Confidence_ML", 
               "Proba_Elo_Player1", "Proba_Elo_Player2", "Predicted_Elo_Winner"]]

if __name__ == "__main__":
    predict_today_matches()
