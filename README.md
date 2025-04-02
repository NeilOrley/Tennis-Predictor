
# 🎾 Tennis Match Predictor

Un projet complet de machine learning pour prédire l'issue de matchs de tennis professionnels (ATP, ITF...).

---

## 📁 Structure du projet

### `fetch_data.py`
- Récupère automatiquement les matchs du jour et les classements depuis Flashscore.
- Génère un fichier `matches_YYYY-MM-DD.csv` dans le dossier `data/`.

### `training.py`
- Entraîne deux modèles :
  - `model_with_odds.pkl` : avec les cotes des joueurs si disponibles.
  - `model_without_odds.pkl` : sans les cotes.
- Exclut désormais les colonnes `Series` et `Best of` pour plus de robustesse.
- Sauvegarde les modèles dans le dossier `models/`.

### `preprocessing.py`
- Contient les fonctions de traitement des données :
  - Enrichissement des features (`H2H`, forme récente, etc.).
  - Chargement des dictionnaires `h2h_dict.pkl` et `recent_form_dict.pkl`.

### `predict_today_matches.py`
- Charge automatiquement les matchs du jour (`data/matches_YYYY-MM-DD.csv`)
- Applique le modèle adapté (`with_odds` ou `without_odds`)
- Produit un fichier `predictions_YYYY-MM-DD.csv` avec :
  - Joueurs
  - Probabilité de victoire (`Proba_Player1`)
  - Gagnant prédit (`Predicted_Winner`)
  - Indice de confiance (`Confidence`)
- Affiche également chaque prédiction en console.

---

## 🧠 Comprendre `Proba` vs `Confiance`

| Champ            | Définition                                                                 |
|------------------|---------------------------------------------------------------------------|
| `Proba_Player1`  | Probabilité estimée que `Player_1` gagne. Ex : 72%                        |
| `Confidence`     | Indice de certitude du modèle. Calcule : `|proba - 0.5| * 200`            |
|                  | Plus la proba est proche de 0% ou 100%, plus la confiance est élevée.     |

### Exemple :

| Proba_Player1 | Gagnant prédit | Confiance |
|---------------|----------------|-----------|
| 0.50          | -              | 0%        |
| 0.75          | Player_1       | 50%       |
| 0.95          | Player_1       | 90%       |
| 0.10          | Player_2       | 80%       |

---

## 🚀 Lancer les prédictions

```bash
python fetch_data.py
python training.py
python predict_today_matches.py
```

---

## 📦 Dossiers attendus

- `data/` → données brutes (matchs, classements)
- `models/` → modèles entraînés
- `predictions/` → prédictions quotidiennes

---

## ✅ TODO
- [ ] Entraîner un modèle qui n'utilise pas les features "Score", "Total_Games" et "Games_Class" pour la prédiction des matchs à venir
- [ ] Ajouter les features avancées sur les derniers
- [ ] Améliorer la logique business (mise, bankroll, ROI, etc.)
- [ ] Intégrer la prédiction du nombre total de jeux
- [ ] Intégrer la prédiction du vainqueur du 1er set
- [ ] Ajouter des prédictions combinées
- [ ] Ajouter une interface Streamlit
- [ ] Intégration continue avec tests unitaires