import argparse
import pandas as pd
from joblib import load
from features import enrich_features, load_h2h_dict, load_recent_form_dict

# 🎯 Chargement des dictionnaires H2H et de forme
h2h_dict = load_h2h_dict("models/h2h_dict.pkl")
form_dict = load_recent_form_dict("models/recent_form_dict.pkl")

# 🎯 Chargement des modèles
model = load("models/tennis_win_predictor.pkl")
games_model = load("models/tennis_total_games_predictor.pkl")

# 🔧 Argument parser
parser = argparse.ArgumentParser(description="Prédiction d'un match de tennis (vainqueur + total de jeux)")

parser.add_argument("--p1", required=True, help="Nom du joueur 1")
parser.add_argument("--p2", required=True, help="Nom du joueur 2")
parser.add_argument("--rank1", type=int, required=True)
parser.add_argument("--rank2", type=int, required=True)
parser.add_argument("--pts1", type=int, required=True)
parser.add_argument("--pts2", type=int, required=True)
parser.add_argument("--odd1", type=float, required=True)
parser.add_argument("--odd2", type=float, required=True)
parser.add_argument("--surface", type=str, required=True)
parser.add_argument("--court", type=str, required=True)
parser.add_argument("--round", type=str, default="1st Round")
parser.add_argument("--bestof", type=int, default=3)
parser.add_argument("--date", type=str, default="2024-03-01", help="Date du match (YYYY-MM-DD)")

args = parser.parse_args()

# 🧾 Construction du DataFrame d'entrée
df_input = pd.DataFrame([{
    "Player_1": args.p1,
    "Player_2": args.p2,
    "Rank_1": args.rank1,
    "Rank_2": args.rank2,
    "Pts_1": args.pts1,
    "Pts_2": args.pts2,
    "Odd_1": args.odd1,
    "Odd_2": args.odd2,
    "Surface": args.surface,
    "Court": args.court,
    "Round": args.round,
    "Best of": args.bestof,
    "Date": pd.to_datetime(args.date)
}])

# 🔧 Enrichissement des features
df_input = enrich_features(df_input, h2h_dict=h2h_dict, recent_form_dict=form_dict)

# 🔮 Prédictions
proba = model.predict_proba(df_input)[0]
games_class = games_model.predict(df_input)[0]

print(f"\n🎾 Match : {args.p1} vs {args.p2}")
print(f"📊 Proba victoire {args.p1} : {proba[1]*100:.1f}%")
print(f"📊 Proba victoire {args.p2} : {proba[0]*100:.1f}%")
print(f"🧮 Intervalle prédit pour le total de jeux : {games_class}")
