import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Setup Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "knowledge_base", "chroma_db")
WHO_URL = "https://www.ncbi.nlm.nih.gov/books/NBK579461/"

def initialize_vector_db():
    embeddings = OllamaEmbeddings(model="nomic-embed-text") 
    
    if not os.path.exists(DB_PATH):
        print(f"Scraping WHO data from {WHO_URL} and building local ChromaDB...")
        loader = WebBaseLoader(WHO_URL)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        
        vectordb = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=DB_PATH)
        vectordb.persist()
        return vectordb
    else:
        print("Loading existing local ChromaDB...")
        return Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def generate_local_treatment(parameter: str, condition: str) -> str:
    vectordb = initialize_vector_db()
    llm = Ollama(model="phi3")
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    
    prompt_template = """
    You are a Senior Water Quality Engineer and Scientist. A water sample has been flagged with a {condition} level of {parameter}.
    
    Use the following pieces of official WHO context to generate a highly detailed, comprehensive, and structured treatment protocol. 
    Do NOT use Roman numerals and do NOT output an index. You must expand on your points deeply and thoroughly.
    If you don't know the answer based on the context, use your deep scientific knowledge, but prioritize the context.
    
    CONTEXT: {context}
    
    You MUST format your response exactly like the template below. You MUST include a blank line between every section.
    
    **Working Principle:**
    [Provide a comprehensive, multi-sentence explanation of the chemical, biological, or physical processes required to treat this specific condition. Explain exactly how and why the treatment works at a technical level.]
    
    **Advantages:**
    - [Provide a detailed explanation of the first major benefit]
    - [Provide a detailed explanation of the second major benefit]
    - [Provide a detailed explanation of the third major benefit]
    
    **Limitations:**
    - [Provide a detailed explanation of the first major drawback, cost, or inefficiency]
    - [Provide a detailed explanation of the second major drawback, cost, or inefficiency]
    - [Provide a detailed explanation of the third major drawback, cost, or inefficiency]
    
    **Precautions:**
    - [Provide a detailed safety, environmental, or monitoring precaution]
    - [Provide a detailed safety, environmental, or monitoring precaution]
    - [Provide a detailed safety, environmental, or monitoring precaution]
    
    Answer:
    """
    prompt = PromptTemplate.from_template(prompt_template)
    
    # Modern LCEL Chain (Replaces the deprecated RetrievalQA)
    rag_chain = (
        {"context": retriever | format_docs, "parameter": RunnablePassthrough(), "condition": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Execute the chain
    response = rag_chain.invoke({"parameter": parameter, "condition": condition})
    return response