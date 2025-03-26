import pandas as pd
from features import enrich_features, load_h2h_dict, load_recent_form_dict
from train import build_preprocessor, train_model
from sklearn.model_selection import train_test_split
import joblib

# Chargement des données
df = pd.read_csv("data/matches_atp_5_dernieres_années.csv")
h2h_dict = load_h2h_dict("models/h2h_dict.pkl")
form_dict = load_recent_form_dict("models/recent_form_dict.pkl")

df = enrich_features(df, h2h_dict=h2h_dict, recent_form_dict=form_dict)
df = df.dropna()

numeric_features = [
    "Rank_1", "Rank_2", "Pts_1", "Pts_2", "Odd_1", "Odd_2", "Best of",
    "Rank_Diff", "Pts_Diff", "Odds_Ratio", "Book_Fav", "Avg_Rank", "Odds_Diff", "Round_Ordinal",
    "H2H_P1", "H2H_P2", "H2H_Diff", "Wins_Last5_P1", "Wins_Last5_P2", "Form_Diff"
]
categorical_features = ["Surface", "Court"]
all_features = numeric_features + categorical_features

df["Winner_encoded"] = (df["Winner"] == df["Player_1"]).astype(int)
X = df[all_features]
y = df["Winner_encoded"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Pipeline + entraînement
preprocessor = build_preprocessor(numeric_features, categorical_features)
model = train_model(X_train, y_train, X_test, y_test, preprocessor)

# Sauvegarde
joblib.dump(model, "models/tennis_win_predictor.pkl")
print("✅ Modèle calibré sauvegardé avec succès.")