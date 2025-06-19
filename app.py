# app.py

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM, OllamaEmbeddings

# --- Config ---
DB_SAVE_PATH = "faiss_index"
MODEL_NAME = "llama3:latest"
BASE_URL = "http://localhost:11434"

st.title("Spam Email Classifier using LLM and Similarity Search")

query = st.text_area("Paste your email content here to classify (spam / not spam):")

if query:

    embedding = OllamaEmbeddings(model=MODEL_NAME, base_url=BASE_URL)
    db = FAISS.load_local(DB_SAVE_PATH, embedding, allow_dangerous_deserialization=True)

    #  top-k similar emails
    query_embedding = embedding.embed_query(query)
    similar_docs = db.similarity_search_by_vector(query_embedding, k=5)

    # Construct LLM prompt
    context = "\n\n".join(
        [f"[Label: {doc.metadata.get('OUTPUT')}]\n{doc.page_content}" for doc in similar_docs]
    )
    prompt = f"""Below are some labeled email examples. Based on their similarity to the input, classify the input as 'spam' or 'not spam' and briefly explain why.

{context}

Input Email:
{query}

Answer:"""

    # Run LLM
    llm = OllamaLLM(model=MODEL_NAME, base_url=BASE_URL)
    response = llm(prompt)

    # Output
    st.subheader("LLM Classification Result:")
    st.write(response)
