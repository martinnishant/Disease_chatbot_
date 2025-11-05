from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib, json
import json

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

# -------------------- Initialize App --------------------
app = FastAPI(title="AI Health Chatbot (ML + RAG + LLM)")

# -------------------- Load ML Model --------------------
clf = joblib.load("models/disease_rf.pkl")
le = joblib.load("models/label_encoder.pkl")
sym_cols = pd.Index(json.loads(open("models/symptom_columns.json").read()))

# -------------------- Load RAG Store --------------------
embedding = OllamaEmbeddings(model="mxbai-embed-large")
db = Chroma(persist_directory="rag_store", embedding_function=embedding)

# -------------------- Input Schema --------------------
class SymptomQuery(BaseModel):
    symptoms: list[str]
    top_n: int | None = 3

# -------------------- Initialize LLM --------------------
llm = ChatOllama(model="llama3.2")  # or "llama3:latest"

# -------------------- Chat Diagnose Endpoint --------------------
@app.post("/chat_diagnose")
def chat_diagnose(data: SymptomQuery):
    # Step 1: Predict using ML
    x = pd.DataFrame([[1 if c in data.symptoms else 0 for c in sym_cols]], columns=sym_cols)
    probs = clf.predict_proba(x)[0]
    idxs = np.argsort(probs)[-data.top_n:][::-1]
    diseases = le.inverse_transform(idxs)
    confs = [float(probs[i]) for i in idxs]

    # Step 2: Retrieve context from RAG
    disease = diseases[0]
    results = db.similarity_search(disease, k=1)
    context = results[0].page_content if results else "No detailed info found."

    # Step 3: Generate natural doctor-style response
    prompt = ChatPromptTemplate.from_template("""
    You are a medical information assistant. 
    Use the context below to explain the possible condition in a structured, factual way.

    Patient symptoms: {symptoms}
    Predicted condition: {disease} (confidence: {confidence})
    Background knowledge:
    {context}

    Return your answer strictly as a valid JSON object with these fields:
    {{
      "summary": "Brief overview of what the condition is.",
      "common_causes": "Usual causes or risk factors.",
      "treatment_and_care": "Typical treatment approaches and lifestyle guidance.",
      "when_to_see_doctor": "Red flags or situations that require medical attention."
    }}
    Avoid extra text outside the JSON.
    """)



    formatted_prompt = prompt.format(
        symptoms=", ".join(data.symptoms),
        disease=disease,
        confidence=round(confs[0], 2),
        context=context
    )

    response = llm.invoke(formatted_prompt)
    try:
        structured = json.loads(response.content)
    except json.JSONDecodeError:
        structured = {"summary": response.content}

    return {"diagnosis": structured}

# -------------------- Health Check --------------------
@app.get("/")
def home():
    return {"message": "AI Doctor Chatbot with ML + RAG + LLM is running 🚀"}
