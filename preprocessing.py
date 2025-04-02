# --- From create_h2h_form_dicts.py ---

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

# --- From features.py ---

import pandas as pd
import numpy as np
from collections import defaultdict, deque
import re
import joblib

def compute_total_games(score_str):
    if pd.isna(score_str):
        return None
    sets = re.findall(r'(\d+)-(\d+)', score_str)
    return sum(int(a) + int(b) for a, b in sets)

def load_h2h_dict(path="models/h2h_dict.pkl"):
    return joblib.load(path)

def load_recent_form_dict(path="models/recent_form_dict.pkl"):
    return joblib.load(path)

def enrich_features(df, h2h_dict=None, recent_form_dict=None):
    df["Rank_Diff"] = df["Rank_1"] - df["Rank_2"]
    df["Pts_Diff"] = df["Pts_1"] - df["Pts_2"]
    df["Avg_Rank"] = (df["Rank_1"] + df["Rank_2"]) / 2

    # Colonnes liées aux cotes (si présentes)
    if "Odd_1" in df.columns and "Odd_2" in df.columns:
        df["Odds_Ratio"] = df["Odd_1"] / df["Odd_2"]
        df["Book_Fav"] = (df["Odd_1"] < df["Odd_2"]).astype(int)
        df["Odds_Diff"] = abs(df["Odd_1"] - df["Odd_2"])

    round_order = {
        "1st Round": 1, "2nd Round": 2, "3rd Round": 3, "4th Round": 4,
        "Quarterfinal": 5, "Semifinal": 6, "Final": 7
    }
    df["Round_Ordinal"] = df["Round"].map(round_order)

    if h2h_dict:
        def get_h2h(p1, p2):
            key = tuple(sorted([p1, p2]))
            h2h = h2h_dict.get(key, [0, 0])
            return h2h if p1 <= p2 else h2h[::-1]

        h2h_values = df.apply(lambda row: get_h2h(row["Player_1"], row["Player_2"]), axis=1)
        df["H2H_P1"] = [val[0] for val in h2h_values]
        df["H2H_P2"] = [val[1] for val in h2h_values]
        df["H2H_Diff"] = df["H2H_P1"] - df["H2H_P2"]

    if recent_form_dict:
        def get_form(player, date):
            if isinstance(date, str):
                date = pd.to_datetime(date, errors='coerce')
            history = recent_form_dict.get(player, [])
            if history and isinstance(history[0], tuple):
                filtered = [res for match_date, res in history if pd.to_datetime(match_date) < date]
                return sum(filtered[-5:])
            else:
                return sum(history[-5:])

        
    
    # Calcul de forme pondérée temporelle
    if "Winner" in df.columns:
        match_history = {}

        # On construit un historique des derniers matchs avec date + résultat
        for idx, row in df.iterrows():
            p1 = row["Player_1"]
            p2 = row["Player_2"]
            date = row["Date"]
            winner = row["Winner"]

            for player, result in [(p1, winner == p1), (p2, winner == p2)]:
                if player not in match_history:
                    match_history[player] = []
                match_history[player].append((date, int(result)))

        df["Form_Score_P1"] = df.apply(lambda row: compute_weighted_form(row["Player_1"], row["Date"], match_history), axis=1)
        df["Form_Score_P2"] = df.apply(lambda row: compute_weighted_form(row["Player_2"], row["Date"], match_history), axis=1)
        df["Form_Diff"] = df["Form_Score_P1"] - df["Form_Score_P2"]
    else:
        df["Form_Score_P1"] = 0.0
        df["Form_Score_P2"] = 0.0
        df["Form_Diff"] = 0.0
    


    return df

def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    if "Score" in df.columns:
        df["Total_Games"] = df["Score"].apply(compute_total_games)
        bins = list(range(15, 40, 4))
        labels = [f"{b}-{b+3}" for b in bins[:-1]]
        df["Games_Class"] = pd.cut(df["Total_Games"], bins=bins, labels=labels, include_lowest=True)
        df["First_Set_Winner"] = df.apply(compute_first_set_winner, axis=1)
        df = df.dropna(subset=["First_Set_Winner"])
    else:
        print("Info : La colonne 'Score' n'est pas présente, ajout de valeurs par défaut pour Total_Games et Games_Class.")
        # On attribue une valeur par défaut (ici 0) pour Total_Games
        df["Total_Games"] = 0
        # Pour Games_Class, on choisit un intervalle par défaut, par exemple "15-18"
        df["Games_Class"] = "15-18"
        # On ne calcule pas First_Set_Winner car le score est absent.
    
    # Transformations communes
    df["Rank_Diff"] = df["Rank_1"] - df["Rank_2"]
    df["Pts_Diff"] = df["Pts_1"] - df["Pts_2"]
    df["Avg_Rank"] = (df["Rank_1"] + df["Rank_2"]) / 2

    if "Odd_1" in df.columns and "Odd_2" in df.columns:
        df["Odds_Ratio"] = df["Odd_1"] / df["Odd_2"]
        df["Book_Fav"] = (df["Odd_1"] < df["Odd_2"]).astype(int)
        df["Odds_Diff"] = abs(df["Odd_1"] - df["Odd_2"])

    round_order = {
        "1st Round": 1, "2nd Round": 2, "3rd Round": 3, "4th Round": 4,
        "Quarterfinal": 5, "Semifinal": 6, "Final": 7
    }
    df["Round_Ordinal"] = df["Round"].map(round_order)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Calcul des H2H et de la forme (exemple)
    h2h_dict = load_h2h_dict("models/h2h_dict.pkl")
    def get_h2h(p1, p2):
        key = tuple(sorted([p1, p2]))
        h2h = h2h_dict.get(key, [0, 0])
        return h2h if p1 <= p2 else h2h[::-1]
    df[["H2H_P1", "H2H_P2"]] = df.apply(lambda row: pd.Series(get_h2h(row["Player_1"], row["Player_2"])), axis=1)
    df["H2H_Diff"] = df["H2H_P1"] - df["H2H_P2"]

    recent_form = load_recent_form_dict("models/recent_form_dict.pkl")
    def get_form(player, date):
        if player not in recent_form:
            return 0
        history = [res for match_date, res in recent_form[player] if pd.to_datetime(match_date) < pd.to_datetime(date)]
        return sum(history[-5:])
    
    # Calcul de forme pondérée temporelle
    if "Winner" in df.columns:
        match_history = {}
        for idx, row in df.iterrows():
            p1 = row["Player_1"]
            p2 = row["Player_2"]
            date = row["Date"]
            winner = row["Winner"]
            for player, result in [(p1, winner == p1), (p2, winner == p2)]:
                if player not in match_history:
                    match_history[player] = []
                match_history[player].append((date, int(result)))
        df["Form_Score_P1"] = df.apply(lambda row: compute_weighted_form(row["Player_1"], row["Date"], match_history), axis=1)
        df["Form_Score_P2"] = df.apply(lambda row: compute_weighted_form(row["Player_2"], row["Date"], match_history), axis=1)
        df["Form_Diff"] = df["Form_Score_P1"] - df["Form_Score_P2"]
    else:
        df["Form_Score_P1"] = 0.0
        df["Form_Score_P2"] = 0.0
        df["Form_Diff"] = 0.0

    return df


from math import exp

def compute_weighted_form(player, match_date, match_history, alpha=0.1):
    if player not in match_history:
        return 0.0

    recent_matches = match_history[player]
    score = 0.0
    for past_date, win in recent_matches:
        days = (match_date - past_date).days
        if days < 0 or days > 180:  # max 6 mois d'historique
            continue
        weight = exp(-alpha * days)
        score += weight * win
    return score

def compute_first_set_winner(row):
    """
    Retourne 1 si le joueur 1 a gagné le premier set, 0 si le joueur 2 l'a gagné.
    Si le score est manquant ou mal formaté, retourne None.
    """
    score_str = row.get("Score")
    if pd.isna(score_str):
        return None
    try:
        # Extraction du premier set (la première partie avant un espace)
        first_set = score_str.split()[0]
        p1_games, p2_games = map(int, first_set.split('-'))
        return 1 if p1_games > p2_games else 0
    except Exception:
        return None



def add_recent_stats(df):
    df = df.sort_values("Date")
    stats = []

    for player_col in ["Player_1", "Player_2"]:
        for stat_name, func in {
            "WinRate_Last5": lambda g: g["win"].rolling(5).mean(),
            "SetWinRate_Last5": lambda g: g["sets_won"] / g["sets_played"],
            "AvgGamesWon_Last5": lambda g: g["games_won"].rolling(5).mean(),
            "AvgGamesDiff_Last5": lambda g: (g["games_won"] - g["games_lost"]).rolling(5).mean(),
            "Matches_Played_Last30D": lambda g: g["Date"].rolling("30D").count(),
        }.items():
            records = []
            grouped = df.groupby(player_col)
            for player, group in grouped:
                group = group.copy()
                group["win"] = (group["Winner"] == group[player_col]).astype(int)
                group["sets_won"] = 2  # simplification
                group["sets_played"] = 3  # simplification
                group["games_won"] = 12  # placeholder
                group["games_lost"] = 10  # placeholder
                group["Date"] = pd.to_datetime(group["Date"])
                group = group.sort_values("Date")
                group[stat_name] = func(group).fillna(0)
                records.append(group[["Date", stat_name]])

            stats_df = pd.concat(records)
            stats_df = stats_df.reset_index(drop=True)

            df = df.merge(stats_df, on="Date", how="left")

    # Crée les versions différentielles
    df["WinRate_Diff"] = df["WinRate_Last5_x"] - df["WinRate_Last5_y"]
    df["AvgGamesDiff_Diff"] = df["AvgGamesDiff_Last5_x"] - df["AvgGamesDiff_Last5_y"]
    df["Matches_Last30D_Diff"] = df["Matches_Played_Last30D_x"] - df["Matches_Played_Last30D_y"]

    return df



def add_recent_stats(df):
    df = df.sort_values("Date").copy()
    
    # Initialisation des colonnes
    for suffix in ["P1", "P2"]:
        df[f"WinRate_Last5_{suffix}"] = 0.0
        df[f"AvgGamesDiff_Last5_{suffix}"] = 0.0
        df[f"Matches_Last30D_{suffix}"] = 0

    # Historique par joueur
    history = {}

    for idx, row in df.iterrows():
        for player_col, suffix in [("Player_1", "P1"), ("Player_2", "P2")]:
            player = row[player_col]
            date = row["Date"]

            if player not in history:
                history[player] = []

            past_matches = [m for m in history[player] if m["Date"] < date]
            recent_5 = past_matches[-5:]
            last_30d = [m for m in past_matches if (date - m["Date"]).days <= 30]

            # Win rate
            wins = [m["Win"] for m in recent_5]
            win_rate = sum(wins) / len(wins) if wins else 0.0

            # Jeu gagné/perdu (simulé)
            games_diff = [m["Games_Won"] - m["Games_Lost"] for m in recent_5]
            avg_diff = sum(games_diff) / len(games_diff) if games_diff else 0.0

            # Nb matchs dans les 30 jours
            n_last_30d = len(last_30d)

            df.at[idx, f"WinRate_Last5_{suffix}"] = win_rate
            df.at[idx, f"AvgGamesDiff_Last5_{suffix}"] = avg_diff
            df.at[idx, f"Matches_Last30D_{suffix}"] = n_last_30d

            # Ajout au log
            history[player].append({
                "Date": date,
                "Win": 1 if row["Winner"] == player else 0,
                "Games_Won": 12,  # simplification
                "Games_Lost": 10   # simplification
            })

    # Colonnes différentielles
    df["WinRate_Diff"] = df["WinRate_Last5_P1"] - df["WinRate_Last5_P2"]
    df["AvgGamesDiff_Diff"] = df["AvgGamesDiff_Last5_P1"] - df["AvgGamesDiff_Last5_P2"]
    df["Matches_Last30D_Diff"] = df["Matches_Last30D_P1"] - df["Matches_Last30D_P2"]

    return df
