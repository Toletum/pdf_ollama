from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_community.vectorstores import FAISS

# Definir la plantilla del prompt
template = """
Tú eres un asistente en español. Usa los documentos que recibes en el contexto para responder en español.
Se breve y conciso
Question: {question}
Context: {context}
Answer:
"""

def main():
    try:
        # Cargar embeddings y modelo
        embeddings = OllamaEmbeddings(base_url="http://toletum:11434", model="deepseek-r1:8b")
        
        # Cargar el vector store localmente
        vector_store = FAISS.load_local("./data", embeddings, allow_dangerous_deserialization=True)
        
        # Inicializar el modelo LLM
        model = OllamaLLM(base_url="http://toletum:11434", model="deepseek-r1:8b")
        
        # Definir la pregunta
        question = "Quién es Jose Carlos?"
        
        # Realizar la búsqueda de similitud en el vector store
        documents = vector_store.similarity_search(question)
        
        # Construir el contexto a partir de los documentos encontrados
        context = "\n\n".join([doc.page_content for doc in documents])
        
        # Crear el prompt a partir de la plantilla
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | model
        
        # Invocar el modelo con la pregunta y el contexto
        answer = chain.invoke({"question": question, "context": context})
        
        # Imprimir la respuesta
        print(answer)
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

