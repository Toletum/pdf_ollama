import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Inicializar embeddings
embeddings = OllamaEmbeddings(base_url="http://toletum:11434", model="deepseek-r1:8b")

pdf_directory = "pdf"
processed_files = set()


try:
    vector_store = FAISS.load_local("./data", embeddings, allow_dangerous_deserialization=True)
    for doc in vector_store.docstore._dict.values():
        if "source" in doc.metadata:
            processed_files.add(doc.metadata["source"])
except Exception as ex:
    vector_store = None


all_documents = []
# Cargar documentos PDF
for filename in os.listdir(pdf_directory):
    if filename.endswith(".pdf") and filename not in processed_files:   
        filepath = os.path.join(pdf_directory, filename)
        print(f"Loading... {filepath}")
        loader = PDFPlumberLoader(filepath)
        documents = loader.load()
        for doc in documents:
            doc.metadata["source"] = filename
        all_documents.extend(documents)


if all_documents:
    # Dividir documentos en fragmentos
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunked_documents = text_splitter.split_documents(all_documents)

    if vector_store:
        print("Vector Database update")
        vector_store.add_documents(chunked_documents)
    else:    
        print("Vector Database create")
        vector_store = FAISS.from_documents(chunked_documents, embedding=embeddings)

    # Guardar la base de datos en disco
    vector_store.save_local("./data")
else:
    print("Nothing to do")
