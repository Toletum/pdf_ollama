import os
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# Inicializar embeddings
embeddings = OllamaEmbeddings(base_url="http://toletum:11434", model="deepseek-r1:8b")

pdf_directory = "pdf"
all_documents = []

vector_store = FAISS.load_local("./data", embeddings, allow_dangerous_deserialization=True)


for doc in vector_store.docstore._dict.values():
    if "source" in doc.metadata:
        print(doc.metadata["source"])

