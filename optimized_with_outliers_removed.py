import pandas as pd
import numpy as np
df = pd.read_csv('Titanic.csv')
print(df)

print(df.isna().sum())

df = df.drop(columns=['deck'])
print(df.isna().sum())

df = df.dropna(subset=['embarked'])
df = df.dropna(subset=['embark_town'])

print(df.isna().sum())

df['age'] = df['age'].fillna(df['age'].median())
print(df.isna().sum())

df['adult_male'] = df['adult_male'].astype(str)
print(df['adult_male'])

df['adult_male'] = df['adult_male'].apply(lambda x: 1 if x == 'True' else 0)

df['alone'] = df['alone'].astype(str)
print(df['alone'])

df['alone'] = df['alone'].apply(lambda x: 1 if x == 'True' else 0)

categorical = ['sex','class','who','embark_town']

numerical = df[['age','adult_male','alone','fare']]

for col in numerical:
    lower = df[col].quantile(0.01)
    upper = df[col].quantile(0.99)
    df[col] = df[col].where((df[col]>=lower) & (df[col]<=upper),np.nan)

from sklearn.preprocessing import LabelEncoder
label_encoders = {}
for col in ['sex','class','who','embark_town']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save for future predictions

x = pd.concat([
    numerical, df[categorical]
],axis='columns')

y = df['survived']

print(x)
print(y)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Logistic Regression (scaled)
log_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000))
])
log_grid = GridSearchCV(
    log_pipe,
    {
        "logreg__C": [0.01, 0.1, 1, 10],
        "logreg__solver": ["lbfgs"],
        "logreg__class_weight": [None,'balanced']
    },
    cv=5,
    scoring="accuracy",
    n_jobs=1
)
log_grid.fit(x_train, y_train)
print("Logistic Regression best params:", log_grid.best_params_)
best_logreg = log_grid.best_estimator_
print('Train R2:',best_logreg.score(x_train,y_train))
print('Test R2:',best_logreg.score(x_test,y_test))

# SVC (scaled)
svc_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC())
])
svc_grid = GridSearchCV(
    svc_pipe,
    {
        "svc__kernel": ["linear", "rbf"],
        "svc__C": [0.1, 1, 10],
        "svc__gamma": ["scale"]
    },
    cv=5,
    scoring="accuracy",
    n_jobs=1
)
svc_grid.fit(x_train, y_train)
print("SVC best params:", svc_grid.best_params_)

best_svc = svc_grid.best_estimator_
print('Train R2:',best_svc.score(x_train,y_train))
print('Test R2:',best_svc.score(x_test,y_test))

# Random Forest Classifier
rfc_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    {
        "n_estimators": [100, 300],
        "max_depth": [None, 10],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 3],
        "max_features": ["sqrt"]
    },
    cv=5,
    scoring="accuracy",
    n_jobs=1
)
rfc_grid.fit(x_train, y_train)
print("Random Forest Classifier best params:", rfc_grid.best_params_)

best_rfc = rfc_grid.best_estimator_
print('Train R2:',best_rfc.score(x_train,y_train))
print('Test R2:',best_rfc.score(x_test,y_test))

y_pred_logreg = best_logreg.predict(x_test)
y_pred_svc = best_svc.predict(x_test)
y_pred_rf = best_rfc.predict(x_test)

from sklearn.metrics import recall_score, accuracy_score, precision_score

print("Logreg Recall:", recall_score(y_test, y_pred_logreg))
print("Logreg Precision:", precision_score(y_test, y_pred_logreg))
print("Logreg Accuracy:", accuracy_score(y_test, y_pred_logreg))

print("svc Recall:", recall_score(y_test, y_pred_svc))
print("svc Precision:", precision_score(y_test, y_pred_svc))
print("svc Accuracy:", accuracy_score(y_test, y_pred_svc))

print("rf Recall:", recall_score(y_test, y_pred_rf))
print("rf Precision:", precision_score(y_test, y_pred_rf))
print("rf Accuracy:", accuracy_score(y_test, y_pred_rf))