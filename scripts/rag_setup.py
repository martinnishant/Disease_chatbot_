import pandas as pd
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# load your disease info
df = pd.read_csv("data/disease_info.csv")

docs = []
for _, row in df.iterrows():
    content = (
        f"Disease: {row['disease']}\n"
        f"Description: {row['description']}\n"
        f"Treatment: {row['treatment']}\n"
        f"Precautions: {row['precautions']}\n"
        f"Home Remedies: {row['home_remedies']}"
    )
    docs.append(Document(page_content=content, metadata={"disease": row["disease"]}))

# use Ollama local embeddings
embedding = OllamaEmbeddings(model="mxbai-embed-large")

db = Chroma.from_documents(docs, embedding, persist_directory="rag_store")
db.persist()
print("✅ RAG store built successfully!")
