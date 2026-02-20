import pandas as pd
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

categorical = ['sex','class','embark_town']

from sklearn.preprocessing import LabelEncoder

label_encoders = {}
for col in ['sex','class','embark_town']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save for future predictions

numerical = df[['age','alone','fare']]

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

y_pred_svc = best_svc.predict(x_test)

from sklearn.metrics import recall_score, accuracy_score, precision_score

print("svc Recall:", recall_score(y_test, y_pred_svc))
print("svc Precision:", precision_score(y_test, y_pred_svc))
print("svc Accuracy:", accuracy_score(y_test, y_pred_svc))

import joblib

joblib.dump(best_svc, 'final_svc_model.pkl')
joblib.dump(x.columns, 'columns.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
print("Model and encoders saved successfully!")