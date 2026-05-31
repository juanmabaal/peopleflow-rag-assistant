from langchain_openai import  OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from backend.config.settings import EMBEDDING_MODEL, PINECONE_INDEX_NAME

def get_embedding():
    return OpenAIEmbeddings()

def get_pinecone_vectore_store():
    embeddings = get_embedding()

    return PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings
    )

def add_documents_to_pinecone(documents):
    vector_store = get_pinecone_vectore_store()
    vector_store.add_documents(documents)
