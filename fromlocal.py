from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_community.vectorstores import FAISS


template = """
Tú eres un asistente en español. Usa los documentos que recibes en el contexto para responder en español.
Se breve y conciso
Question: {question}
Context: {context}
Answer:
"""

embeddings = OllamaEmbeddings(base_url="http://toletum:11434", model="deepseek-r1:8b")

vector_store = FAISS.load_local("./data", embeddings, allow_dangerous_deserialization=True)

model = OllamaLLM(base_url="http://toletum:11434", model="deepseek-r1:8b")

question = "Quién es Jose Carlos?"

documents = vector_store.similarity_search(question)

context = "\n\n".join([doc.page_content for doc in documents])
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

answer = chain.invoke({"question": question, "context": context})

print(answer)

