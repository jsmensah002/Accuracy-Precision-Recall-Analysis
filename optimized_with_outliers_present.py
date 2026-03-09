# Always check for class imbalance and confusion matrix!!!!!

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

df = pd.read_csv('breast_cancer.csv')
print(df)

print(df.isna().sum())
print(df.duplicated().sum())

df['diagnosis'] = df['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)

columns = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
           'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean',
           'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
           'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
           'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
           'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst',
           'symmetry_worst', 'fractal_dimension_worst']

x = df[columns]
y = df['diagnosis']

print(x)
print(y)

# --- CLASS IMBALANCE CHECK ---
print("Class Distribution:")
print(df['diagnosis'].value_counts())
print(df['diagnosis'].value_counts(normalize=True) * 100)

# --- TRAIN TEST SPLIT ---
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# --- GRIDSEARCH ---
# LogReg and SVC: scaling inside Pipeline
# RFC and XGB: no scaling needed

# Logistic Regression (scaling inside pipeline)
log_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000))
])
log_grid = GridSearchCV(
    log_pipe,
    {
        "logreg__C": [0.01, 0.1, 1, 10],
        "logreg__solver": ["lbfgs"],
        "logreg__class_weight": [None, 'balanced']
    },
    cv=5, scoring="accuracy", n_jobs=1
)
log_grid.fit(x_train, y_train)
print("Logistic Regression best params:", log_grid.best_params_)
best_logreg = log_grid.best_estimator_

# SVC (scaling inside pipeline)
svc_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(probability=True))
])
svc_grid = GridSearchCV(
    svc_pipe,
    {
        "svc__kernel": ["linear", "rbf"],
        "svc__C": [0.1, 1, 10],
        "svc__gamma": ["scale"]
    },
    cv=5, scoring="accuracy", n_jobs=1
)
svc_grid.fit(x_train, y_train)
print("SVC best params:", svc_grid.best_params_)
best_svc = svc_grid.best_estimator_

# Random Forest (no scaling)
rfc_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    {
        "n_estimators": [100, 300],
        "max_depth": [None, 10],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 3],
        "max_features": ["sqrt"]
    },
    cv=5, scoring="accuracy", n_jobs=1
)
rfc_grid.fit(x_train, y_train)
print("Random Forest best params:", rfc_grid.best_params_)
best_rfc = rfc_grid.best_estimator_

# XGBoost (no scaling)
xgb_grid = GridSearchCV(
    XGBClassifier(random_state=42, eval_metric='logloss'),
    {
        "n_estimators": [100, 300],
        "max_depth": [3, 6, 10],
        "learning_rate": [0.01, 0.1, 0.3],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    },
    cv=5, scoring="accuracy", n_jobs=1
)
xgb_grid.fit(x_train, y_train)
print("XGBoost best params:", xgb_grid.best_params_)
best_xgb = xgb_grid.best_estimator_

# --- EVALUATION ---
models = {
    'LogReg': best_logreg,
    'SVC':    best_svc,
    'RFC':    best_rfc,
    'XGB':    best_xgb
}

best_preds  = {}
best_probas = {}

for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred  = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]
    best_preds[name]  = y_pred
    best_probas[name] = y_proba

    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"{name} Train Score: {model.score(x_train, y_train):.4f}")
    print(f"{name} Test Score:  {model.score(x_test, y_test):.4f}")
    print(f"\n{name} Classification Report:\n{classification_report(y_test, y_pred)}")
    print(f"{name} AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")