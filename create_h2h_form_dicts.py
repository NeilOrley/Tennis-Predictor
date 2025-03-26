import pandas as pd
import joblib
from collections import defaultdict, deque

# Charger les données complètes
df = pd.read_csv("data/atp_tennis.csv")
df = df.dropna(subset=["Player_1", "Player_2", "Winner", "Date"])
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"])
df = df.sort_values("Date")

# 🧠 H2H dict
h2h_dict = defaultdict(lambda: [0, 0])
for _, row in df.iterrows():
    p1, p2, winner = row["Player_1"], row["Player_2"], row["Winner"]
    key = tuple(sorted([p1, p2]))
    if winner == p1:
        h2h_dict[key][0] += 1
    elif winner == p2:
        h2h_dict[key][1] += 1

# 💪 Forme récente dict (date, win)
recent_form_dict = defaultdict(lambda: deque(maxlen=5))
form_history = defaultdict(list)

for _, row in df.iterrows():
    p1, p2, winner, date = row["Player_1"], row["Player_2"], row["Winner"], row["Date"]

    form_history[p1].append((date, 1 if winner == p1 else 0))
    form_history[p2].append((date, 1 if winner == p2 else 0))

# Regroupe les derniers résultats pour chaque joueur
for player, results in form_history.items():
    recent_form_dict[player] = deque(results, maxlen=5)

# 💾 Sauvegarde
joblib.dump(dict(h2h_dict), "models/h2h_dict.pkl")
joblib.dump(dict(recent_form_dict), "models/recent_form_dict.pkl")

print("✅ Dictionnaires enregistrés dans models/")