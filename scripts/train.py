import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("data/Training.csv")
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.drop_duplicates().reset_index(drop=True)
X = df.drop(columns=["prognosis"])
y = df["prognosis"]

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_val, y_train, y_val = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

clf = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print("VAL_ACC", round(acc, 4))
print(classification_report(y_val, y_pred, target_names=le.classes_))

joblib.dump(clf, "models/disease_rf.pkl")
joblib.dump(le, "models/label_encoder.pkl")
X.columns.to_series().to_json("models/symptom_columns.json", orient="values")
print("SAVED models/disease_rf.pkl, models/label_encoder.pkl, models/symptom_columns.json")
