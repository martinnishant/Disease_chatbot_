from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib, json, numpy as np

app = FastAPI(title="AI Disease Chatbot")

# Load model and metadata
clf = joblib.load("models/disease_rf.pkl")
le = joblib.load("models/label_encoder.pkl")
sym_cols = pd.Index(json.loads(open("models/symptom_columns.json").read()))

class Symptoms(BaseModel):
    symptoms: list[str]
    top_n: int | None = 3

@app.get("/")
def home():
    return {"message": "AI Disease Chatbot API is running!"}

@app.post("/predict")
def predict(data: Symptoms):
    x = pd.DataFrame([[1 if c in data.symptoms else 0 for c in sym_cols]], columns=sym_cols)
    probs = clf.predict_proba(x)[0]
    idxs = np.argsort(probs)[-data.top_n:][::-1]
    labels = le.inverse_transform(idxs).tolist()
    confs = [float(probs[i]) for i in idxs]
    return {"predictions": [{"disease": l, "confidence": round(c, 4)} for l, c in zip(labels, confs)]}
