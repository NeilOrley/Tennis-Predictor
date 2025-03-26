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

        df["Wins_Last5_P1"] = df.apply(lambda row: get_form(row["Player_1"], row["Date"]), axis=1)
        df["Wins_Last5_P2"] = df.apply(lambda row: get_form(row["Player_2"], row["Date"]), axis=1)
        df["Form_Diff"] = df["Wins_Last5_P1"] - df["Wins_Last5_P2"]

    return df

def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    df["Total_Games"] = df["Score"].apply(compute_total_games)
    bins = list(range(15, 40, 4))
    labels = [f"{b}-{b+3}" for b in bins[:-1]]
    df["Games_Class"] = pd.cut(df["Total_Games"], bins=bins, labels=labels, include_lowest=True)

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
    df["Wins_Last5_P1"] = df.apply(lambda row: get_form(row["Player_1"], row["Date"]), axis=1)
    df["Wins_Last5_P2"] = df.apply(lambda row: get_form(row["Player_2"], row["Date"]), axis=1)
    df["Form_Diff"] = df["Wins_Last5_P1"] - df["Wins_Last5_P2"]
    return df
