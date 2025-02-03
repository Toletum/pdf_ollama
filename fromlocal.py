from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore


template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
"""

embeddings = OllamaEmbeddings(base_url="http://toletum:11434", model="deepseek-r1:8b")


index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore= InMemoryDocstore(),
    index_to_docstore_id={}
)

vector_store = FAISS.load_local("./data", embeddings, allow_dangerous_deserialization=True)
    
model = OllamaLLM(base_url="http://toletum:11434", model="deepseek-r1:8b")


question = "Who is Jose Carlos?"

documents = vector_store.similarity_search(question)

context = "\n\n".join([doc.page_content for doc in documents])
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

answer = chain.invoke({"question": question, "context": context})

print(answer)

