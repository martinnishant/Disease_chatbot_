import pandas as pd, joblib, json
from sklearn.metrics import accuracy_score

clf = joblib.load("models/disease_rf.pkl")
le = joblib.load("models/label_encoder.pkl")

test = pd.read_csv("data/Testing.csv")
test = test.loc[:, ~test.columns.str.contains('^Unnamed')]  # <-- add this line

X_test = test.drop(columns=["prognosis"])
y_true = test["prognosis"]

y_pred_enc = clf.predict(X_test)
y_pred = le.inverse_transform(y_pred_enc)

print("TEST_ACC", round(accuracy_score(y_true, y_pred), 4))
