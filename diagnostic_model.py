
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from preprocessing import enrich_features, load_h2h_dict, load_recent_form_dict
from collections import Counter

# Chargement du modèle
model_path = "models/model_without_odds.pkl"
model = joblib.load(model_path)

# Chargement des données
df = pd.read_csv("data/atp_tennis.csv")
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Winner_encoded"] = (df["Winner"] == df["Player_1"]).astype(int)

# Filtrage joueurs actifs
player_counts = Counter(df["Player_1"]) + Counter(df["Player_2"])
df = df[df["Player_1"].map(player_counts) >= 5]
df = df[df["Player_2"].map(player_counts) >= 5]

# Enrichissement
h2h_dict = load_h2h_dict()
form_dict = load_recent_form_dict()
df = enrich_features(df, h2h_dict, form_dict)

# Sélection des colonnes
X = df.drop(columns=["Winner", "Winner_encoded", "Player_1", "Player_2", "Score", "Date", "Series", "Best of"], errors="ignore")
y = df["Winner_encoded"]

# Probabilités prédictes
classes = model.classes_
if list(classes) == [0, 1]:
    proba = model.predict_proba(X)[:, 1]
elif list(classes) == [1, 0]:
    proba = model.predict_proba(X)[:, 0]
else:
    raise ValueError(f"Classes inattendues : {classes}")

# Histogramme des probabilités
plt.figure(figsize=(8, 5))
plt.hist(proba, bins=30, color='blue', alpha=0.7)
plt.title("Distribution des probabilités (Proba Player_1)")
plt.xlabel("Probabilité que Player_1 gagne")
plt.ylabel("Nombre de matchs")
plt.grid(True)
plt.tight_layout()
plt.savefig("proba_distribution.png")
plt.close()

# Affichage importance des features si possible
try:
    clf = model.calibrated_classifiers_[0].estimator.named_steps["clf"]
    preproc = model.calibrated_classifiers_[0].estimator.named_steps["preprocessor"]
    feature_names = preproc.get_feature_names_out()
    importances = clf.feature_importances_

    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=True)
    feat_imp[-30:].plot(kind="barh", figsize=(10, 8))
    plt.title("Importance des 30 principales variables")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
except Exception as e:
    print("Impossible d'extraire les importances : ", e)
