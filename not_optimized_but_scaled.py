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

categorical = ['sex','class','who','embark_town']

numerical = df[['age','adult_male','alone','fare']]

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

print(df.select_dtypes(include='number').corr()['survived'].sort_values(ascending=False))

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

logreg = LogisticRegression()
svc = SVC()
rf = RandomForestClassifier(random_state=42)

logreg.fit(x,y)
svc.fit(x,y)
rf.fit(x,y)

from sklearn.model_selection import train_test_split as tt
x_train,x_test,y_train,y_test = tt(x,y,test_size=0.2,random_state=42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

logreg.fit(x_train_scaled,y_train)
svc.fit(x_train_scaled,y_train)

print('Logreg train score:', logreg.score(x_train_scaled,y_train))
print('Logreg test score:',logreg.score(x_test_scaled,y_test))

print('svc train score:',svc.score(x_train_scaled,y_train))
print('svc test score:',svc.score(x_test_scaled,y_test))

y_pred_logreg = logreg.predict(x_test_scaled)
y_pred_svc = svc.predict(x_test_scaled)

from sklearn.metrics import recall_score, accuracy_score, precision_score

print("Logreg Recall:", recall_score(y_test, y_pred_logreg))
print("Logreg Precision:", precision_score(y_test, y_pred_logreg))
print("Logreg Accuracy:", accuracy_score(y_test, y_pred_logreg))

print("svc Recall:", recall_score(y_test, y_pred_svc))
print("svc Precision:", precision_score(y_test, y_pred_svc))
print("svc Accuracy:", accuracy_score(y_test, y_pred_svc))

