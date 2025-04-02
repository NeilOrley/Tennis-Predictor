import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import f1_score, log_loss, roc_auc_score, brier_score_loss
from collections import Counter
from xgboost import XGBClassifier
from preprocessing import enrich_features, compute_all_features, load_h2h_dict, load_recent_form_dict
from preprocessing import add_recent_stats

def compute_match_winner_label(df):
    return (df["Winner"] == df["Player_1"]).astype(int)

def train_model(optimize_hyperparams=True):
    df = pd.read_csv("data/atp_tennis.csv")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    h2h_dict = load_h2h_dict()
    form_dict = load_recent_form_dict()
    df = enrich_features(df, h2h_dict, form_dict)
    df = compute_all_features(df)
    df = add_recent_stats(df)

    df = df.dropna(subset=["Winner"])
    df["Match_Winner_Label"] = compute_match_winner_label(df)

    target = "Match_Winner_Label"
    drop_cols = ["Winner", "Match_Winner_Label", "Player_1", "Player_2", "Score", "Date", "Series", "Best of", "First_Set_Winner"]
    features = [col for col in df.columns if col not in drop_cols and col in df.columns]

    X = df[features]
    y = df[target].astype(int)

    min_class_count = min(Counter(y).values())
    stratify_option = y if min_class_count >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=stratify_option, random_state=42)

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(eval_metric="logloss", random_state=42))
    ])

    if optimize_hyperparams:
        param_dist = {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [3, 5, 7, 10],
            "classifier__learning_rate": [0.01, 0.1, 0.3],
            "classifier__subsample": [0.6, 0.8, 1.0],
            "classifier__colsample_bytree": [0.6, 0.8, 1.0]
        }
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_dist,
            n_iter=20,
            scoring="neg_log_loss",
            cv=5,
            verbose=1,
            random_state=42,
            n_jobs=-1
        )
        search.fit(X_train, y_train)
        model = CalibratedClassifierCV(search.best_estimator_, method="sigmoid", cv=5)
    else:
        model = CalibratedClassifierCV(pipeline, method="sigmoid", cv=5)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred)
    logloss = log_loss(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    brier = brier_score_loss(y_test, y_proba)

    print(f"F1-score   : {f1:.4f}")
    print(f"Log-loss   : {logloss:.4f}")
    print(f"AUC        : {auc:.4f}")
    print(f"Brier score: {brier:.4f}")

    return model

if __name__ == "__main__":
    train_model()
