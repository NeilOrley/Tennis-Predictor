import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer

from features import compute_all_features

# Chargement des données d'entraînement
df = pd.read_csv("data/matches_atp_5_dernieres_années.csv")
df = compute_all_features(df)

# Filtrage des lignes valides pour la prédiction des intervalles de jeux
df = df.dropna(subset=["Games_Class"])

# Définition des features
numeric_features = [
    'Rank_1', 'Rank_2', 'Pts_1', 'Pts_2', 'Odd_1', 'Odd_2', 'Best of',
    'Rank_Diff', 'Pts_Diff', 'Odds_Ratio', 'Book_Fav', 'Avg_Rank',
    'Odds_Diff', 'Round_Ordinal',
    'H2H_P1', 'H2H_P2', 'H2H_Diff',
    'Wins_Last5_P1', 'Wins_Last5_P2', 'Form_Diff'
]
categorical_features = ['Surface', 'Court']
all_features = numeric_features + categorical_features

X = df[all_features]
y = df["Games_Class"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Pipeline
numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean'))])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("clf", GradientBoostingClassifier(random_state=42))
])

param_grid = {
    "clf__n_estimators": [100],
    "clf__max_depth": [3, 5],
    "clf__learning_rate": [0.05, 0.1]
}

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid.fit(X_train, y_train)

print("✅ Meilleur modèle pour la prédiction du nombre de jeux :", grid.best_params_)

# Sauvegarde du modèle
joblib.dump(grid.best_estimator_, "models/tennis_total_games_predictor.pkl")
print("💾 Modèle sauvegardé dans models/tennis_total_games_predictor.pkl")