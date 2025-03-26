# 🎾 Tennis Match Predictor

Ce projet utilise des modèles de machine learning pour prédire l'issue d'un match de tennis professionnel (ATP) à partir de statistiques, de cotes, de confrontations passées (H2H), et de la forme récente des joueurs.

---

## 📦 Contenu

- `features.py` – Calcul des features, H2H, forme récente
- `train.py` – Pipeline d'entraînement et calibration (isotonic/sigmoid)
- `predict.py` – Fonction pour prédire à partir d'un input enrichi
- `main.py` – Script principal pour entraîner et sauvegarder un modèle
- `create_h2h_form_dicts.py` – Génère les dictionnaires H2H et forme récente
- `predict_cli.py` – Interface ligne de commande pour prédire un match
- `playground.ipynb` – Notebook interactif d'exemple
- `models/` – Contiendra les fichiers `.pkl` générés (modèles et dictionnaires)
- `data/` – Doit contenir les fichiers CSV `atp_tennis.csv` et `matches_atp_5_dernieres_années.csv`

---

## 🚀 Installation

```bash
git clone https://github.com/votre-nom/tennis-predictor.git
cd Tennis-Predictor
pip install -r requirements.txt
```

---

## 🔧 Préparation

1. Place tes fichiers CSV dans `./data` :
   - `atp_tennis.csv` – Historique complet ATP
   - `matches_atp_5_dernieres_années.csv` – Données d’entraînement

2. Génère les dictionnaires H2H et forme récente :
```bash
python create_h2h_form_dicts.py
```

3. Entraîne et sauvegarde le modèle :
```bash
python main.py
```

---

## 🧠 Prédire un match (CLI)

```bash
python predict_cli.py --p1 "Djokovic N." --p2 "Alcaraz C." --rank1 1 --rank2 2 \
--pts1 9000 --pts2 8500 --odd1 1.9 --odd2 2.1 --surface "Hard" --court "Outdoor"
```

---

## 🧪 Exemple interactif

Ouvre `playground.ipynb` pour tester en notebook.

---

## 📁 Exemple d’organisation

```
tennis-predictor/
├── data/
│   ├── atp_tennis.csv
│   └── matches_atp_5_dernieres_années.csv
├── models/
│   ├── tennis_win_predictor.pkl
│   ├── h2h_dict.pkl
│   └── recent_form_dict.pkl
├── features.py
├── train.py
├── predict.py
├── main.py
├── predict_cli.py
├── create_h2h_form_dicts.py
├── playground.ipynb
└── README.md
```

---

## 📌 Auteurs

Développé avec ❤️ par [votre nom] – basé sur les travaux du papier ["Machine Learning Techniques for Predicting Tennis Match Outcomes"](https://arxiv.org/pdf/1701.08055).