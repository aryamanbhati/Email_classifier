# generate_db.py

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain.docstore.document import Document
import os

# --- Config ---
CSV_PATH = "spam.csv"         # Your input file
DB_SAVE_PATH = "faiss_index"    # Folder to save FAISS index
MODEL_NAME = "llama3:latest"
BASE_URL = "http://localhost:11434"

# --- Load Data ---
df = pd.read_csv(CSV_PATH, encoding="ISO-8859-1")

if 'TEXT' not in df.columns or 'OUTPUT' not in df.columns:
    raise ValueError("CSV must contain 'TEXT' and 'OUTPUT' columns.")

# --- Convert to Documents ---
documents = [
    Document(page_content=row['TEXT'], metadata={"OUTPUT": row['OUTPUT']})
    for _, row in df.iterrows()
]

# --- Embed and Save ---
embedding = OllamaEmbeddings(model=MODEL_NAME, base_url=BASE_URL)
db = FAISS.from_documents(documents, embedding)
db.save_local(DB_SAVE_PATH)

print(f"✅ FAISS DB created and saved to: {DB_SAVE_PATH}")
