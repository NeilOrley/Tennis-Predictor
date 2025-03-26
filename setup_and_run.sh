#!/bin/bash

echo "🚀 Création d'un environnement virtuel..."
python -m venv tennis_env
source tennis_env/bin/activate

echo "📦 Installation des dépendances..."
pip install --upgrade pip
pip install -r requirements.txt

echo "⚙️ Génération des dictionnaires H2H et Forme récente..."
python create_h2h_form_dicts.py

echo "🧠 Entraînement du modèle..."
python main.py
echo "🎯 Entraînement du modèle d'intervalle de jeux..."
python train_total_games.py

echo "✅ Prêt à prédire !"
echo "Utilise : python predict_cli.py --help pour voir les options"