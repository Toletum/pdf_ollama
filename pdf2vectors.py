from langchain_community.document_loaders import PDFPlumberLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

# Initialize embeddings and vector store
embeddings = OllamaEmbeddings(base_url="http://toletum:11434", model="deepseek-r1:8b")
index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))  # Get embedding dimension dynamically
vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),  # Consider persistent docstore for larger datasets
    index_to_docstore_id={}
)

# Load and chunk the PDF
loader = PDFPlumberLoader("pdf/JoseCarlosSanchezGomezES.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)
chunked_documents = text_splitter.split_documents(documents)

# Add documents to the vector store
vector_store.add_documents(chunked_documents)

# Save the vector store
vector_store.save_local("./data")
