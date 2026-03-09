# Always check for class imbalance and confusion matrix!!!!!

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

df = pd.read_csv('breast_cancer.csv')
print(df)

print(df.isna().sum())
print(df.duplicated().sum())

df['diagnosis'] = df['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)

x = df.drop(columns=['diagnosis', 'id', 'Unnamed: 32'])
y = df['diagnosis']

print(x)
print(y)

# --- CLASS IMBALANCE CHECK ---
print("Class Distribution:")
print(df['diagnosis'].value_counts())
print(df['diagnosis'].value_counts(normalize=True) * 100)

# --- TRAIN TEST SPLIT ---
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

models = {
    'LogReg': LogisticRegression(random_state=42, max_iter=1000),
    'SVC':    SVC(random_state=42, probability=True),
    'RFC':    RandomForestClassifier(random_state=42),
    'XGB':    XGBClassifier(random_state=42)
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