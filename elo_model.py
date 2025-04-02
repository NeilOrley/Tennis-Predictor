# elo_model.py
import pandas as pd
import numpy as np

class EloModel:
    def __init__(self, initial_rating=1500, k=32, factor=400):
        """
        initial_rating : note de départ pour chaque joueur
        k : coefficient d'actualisation (plus grand => plus de volatilité)
        factor : facteur de normalisation (souvent 400 pour le système Elo)
        """
        self.ratings = {}
        self.initial_rating = initial_rating
        self.k = k
        self.factor = factor

    def get_rating(self, player):
        """Retourne le rating actuel d'un joueur, ou le rating initial s'il n'existe pas encore."""
        return self.ratings.get(player, self.initial_rating)

    def prob_victoire(self, rating_A, rating_B):
        """Calcule la probabilité que le joueur A batte le joueur B."""
        return 1 / (1 + 10 ** ((rating_B - rating_A) / self.factor))

    def update_ratings(self, player_A, player_B, result):
        """
        Met à jour les ratings suite à un match.
        result : 1 si player_A gagne, 0 si player_B gagne.
        """
        rating_A = self.get_rating(player_A)
        rating_B = self.get_rating(player_B)
        prob_A = self.prob_victoire(rating_A, rating_B)
        rating_A_new = rating_A + self.k * (result - prob_A)
        rating_B_new = rating_B + self.k * ((1 - result) - (1 - prob_A))
        self.ratings[player_A] = rating_A_new
        self.ratings[player_B] = rating_B_new

    def predict_match(self, player_A, player_B):
        """Retourne la probabilité que le joueur A gagne face au joueur B."""
        rating_A = self.get_rating(player_A)
        rating_B = self.get_rating(player_B)
        return self.prob_victoire(rating_A, rating_B)

def update_elo_from_matches(df, elo_model):
    """
    Met à jour le modèle Elo à partir d'un DataFrame historique.
    Le DataFrame doit contenir les colonnes "Player_1", "Player_2" et "Winner".
    """
    for _, row in df.iterrows():
        player_A = row["Player_1"]
        player_B = row["Player_2"]
        winner = row["Winner"]
        result = 1 if winner == player_A else 0
        elo_model.update_ratings(player_A, player_B, result)
