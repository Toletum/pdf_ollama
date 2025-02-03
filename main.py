from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
"""

embeddings = OllamaEmbeddings(model="deepseek-r1:8b")
vector_store = InMemoryVectorStore(embeddings)

model = OllamaLLM(model="deepseek-r1:8b")


loader = PDFPlumberLoader("pdf/JoseCarlosSanchezGomezES.pdf")
documents = loader.load()


text_splitter = text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
chunked_documents = text_splitter.split_documents(documents)


vector_store.add_documents(chunked_documents)


question = "Who is Jose Carlos?"

vector_store.similarity_search(question)


context = "\n\n".join([doc.page_content for doc in documents])
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

answer = chain.invoke({"question": question, "context": context})

print(answer)

