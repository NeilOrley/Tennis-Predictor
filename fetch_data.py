# --- From fetch_matches_today.py ---

import requests
import os
import pandas as pd
from datetime import datetime
import urllib3

# ⚠️ Désactiver les avertissements SSL (temporaire uniquement)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def fetch_flashscore_rankings(save_dir="data", force_reload=False):
    """
    Télécharge et enregistre le classement ATP Singles depuis Flashscore API.
    """
    os.makedirs(save_dir, exist_ok=True)

    today_str = datetime.today().strftime("%Y-%m-%d")
    filename = f"rankings_atp_{today_str}.csv"
    filepath = os.path.join(save_dir, filename)

    if os.path.exists(filepath) and not force_reload:
        print(f"✅ Classement déjà existant : {filepath}")
        return pd.read_csv(filepath)

    url_rankings_list = "https://flashlive-sports.p.rapidapi.com/v1/rankings/list?sport_id=2&locale=en_INT"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "x-rapidapi-key": "f4ba8f88b9msh4cda0a37a7df344p16b553jsn382473e118a4",
        "x-rapidapi-host": "flashlive-sports.p.rapidapi.com"
    }

    response = requests.get(url_rankings_list, headers=headers, verify=False)
    response.raise_for_status()
    rankings_list = response.json()

    # Récupérer l’ID du ranking ATP Singles
    ranking_id = next((r["RANKING_ID"] for r in rankings_list["DATA"] if "ATP Singles" in r["RANKING_LABEL"]), None)
    if not ranking_id:
        raise ValueError("Classement ATP Singles introuvable.")

    # Récupération des joueurs classés
    url_data = f"https://flashlive-sports.p.rapidapi.com/v1/rankings/data?ranking_id={ranking_id}&locale=en_INT"
    response = requests.get(url_data, headers=headers, verify=False)
    response.raise_for_status()
    players = response.json().get("DATA", [])

    df = pd.DataFrame(players)[["RANK", "PARTICIPANT_NAME", "PARTICIPANT_ID", "RESULT"]]
    df.columns = ["Rank", "Player", "Player_ID", "Points"]
    df["Rank"] = df["Rank"].str.replace(".", "").astype(int)
    df["Points"] = pd.to_numeric(df["Points"], errors="coerce")

    df.to_csv(filepath, index=False)
    print(f"✅ Classement ATP sauvegardé dans : {filepath}")
    return df


def fetch_flashscore_matches(save_dir="data", force_reload=False):
    """
    Récupère les matchs du jour (hors doubles), intègre le ranking des joueurs et sauvegarde dans un fichier CSV.
    """
    os.makedirs(save_dir, exist_ok=True)
    today_str = datetime.today().strftime("%Y-%m-%d")
    filename = f"matches_{today_str}.csv"
    filepath = os.path.join(save_dir, filename)

    if os.path.exists(filepath) and not force_reload:
        print(f"✅ Matchs déjà récupérés : {filepath}")
        return pd.read_csv(filepath)

    # Charger le classement
    rankings_df = fetch_flashscore_rankings(save_dir=save_dir)

    # Requête matches
    url = "https://flashlive-sports.p.rapidapi.com/v1/events/list?locale=en_INT&timezone=-4&sport_id=2&indent_days=0"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "x-rapidapi-key": "f4ba8f88b9msh4cda0a37a7df344p16b553jsn382473e118a4",
        "x-rapidapi-host": "flashlive-sports.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, verify=False)
    response.raise_for_status()
    data = response.json()

    rows = []
    today = pd.to_datetime("today").normalize()

    for tournament in data.get("DATA", []):
        if "Doubles" in tournament.get("NAME", ""):
            continue

        surface = tournament["NAME"].split(",")[-1].strip()
        tournament_name = tournament.get("SHORT_NAME", "")
        events = tournament.get("EVENTS", [])

        for match in events:
            try:
                id_p1 = match["HOME_PARTICIPANT_IDS"][0]
                id_p2 = match["AWAY_PARTICIPANT_IDS"][0]

                row = {
                    "Tournament": tournament_name,
                    "Surface": surface,
                    "Court": "Outdoor",
                    "Player_1": match["HOME_PARTICIPANT_NAME_ONE"],
                    "Player_2": match["AWAY_PARTICIPANT_NAME_ONE"],
                    "ID_P1": id_p1,
                    "ID_P2": id_p2,
                    "Round": match.get("ROUND", None),
                    "Date": today,
                }

                row["EVENT_ID"] = match.get("EVENT_ID")
                if row["EVENT_ID"]:
                    odd1, odd2 = fetch_odds_for_event(row["EVENT_ID"])
                    row["Odd_1"] = odd1
                    row["Odd_2"] = odd2
                else:
                    row["Odd_1"] = None
                    row["Odd_2"] = None

                # Ajout des rankings si disponibles
                rank_p1 = rankings_df.loc[rankings_df["Player_ID"] == id_p1]
                rank_p2 = rankings_df.loc[rankings_df["Player_ID"] == id_p2]

                row["Rank_1"] = int(rank_p1["Rank"].values[0]) if not rank_p1.empty else None
                row["Rank_2"] = int(rank_p2["Rank"].values[0]) if not rank_p2.empty else None
                row["Pts_1"] = int(rank_p1["Points"].values[0]) if not rank_p1.empty else None
                row["Pts_2"] = int(rank_p2["Points"].values[0]) if not rank_p2.empty else None

                rows.append(row)

            except Exception as e:
                print(f"❌ Erreur match : {e}")
                continue

    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    print(f"✅ Matchs sauvegardés dans : {filepath}")
    return df


def fetch_odds_for_event(event_id):
    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "x-rapidapi-key": "f4ba8f88b9msh4cda0a37a7df344p16b553jsn382473e118a4",
        "x-rapidapi-host": "flashlive-sports.p.rapidapi.com"
    }

    url = f"https://flashlive-sports.p.rapidapi.com/v1/events/odds?event_id={event_id}&locale=en_INT"

    try:
        response = requests.get(url, headers=headers, verify=False)
        response.raise_for_status()
        data = response.json()

        all_odds_1 = []
        all_odds_2 = []

        for block in data.get("DATA", []):
            if block.get("BETTING_TYPE") == "*Home/Away":
                for period in block.get("PERIODS", []):
                    if period.get("ODDS_STAGE") == "*Match":
                        for group in period.get("GROUPS", []):
                            for market in group.get("MARKETS", []):
                                odd1 = float(market.get("ODD_CELL_SECOND", {}).get("VALUE", 0))
                                odd2 = float(market.get("ODD_CELL_THIRD", {}).get("VALUE", 0))
                                if odd1 > 0:
                                    all_odds_1.append(odd1)
                                if odd2 > 0:
                                    all_odds_2.append(odd2)

        if all_odds_1 and all_odds_2:
            return max(all_odds_1), max(all_odds_2)
    except Exception as e:
        print(f"[⚠️] Erreur cotes pour event {event_id}: {e}")

    return None, None


if __name__ == "__main__":
    df_matches = fetch_flashscore_matches()
    print(df_matches.head())