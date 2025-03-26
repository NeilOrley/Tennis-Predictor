import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss

def build_preprocessor(numeric_features, categorical_features):
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    return ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

def train_model(X_train, y_train, X_test, y_test, preprocessor):
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", LogisticRegression(max_iter=2000, solver='saga', random_state=42))
    ])
    pipe.fit(X_train, y_train)

    cal_iso = CalibratedClassifierCV(pipe, method="isotonic", cv=3)
    cal_iso.fit(X_train, y_train)
    brier_iso = brier_score_loss(y_test, cal_iso.predict_proba(X_test)[:, 1])

    cal_sig = CalibratedClassifierCV(pipe, method="sigmoid", cv=3)
    cal_sig.fit(X_train, y_train)
    brier_sig = brier_score_loss(y_test, cal_sig.predict_proba(X_test)[:, 1])

    return cal_iso if brier_iso < brier_sig else cal_sig