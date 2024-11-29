import chromadb
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()

class CustomOpenAIEmbeddings(OpenAIEmbeddings):
    def __init__(self, openai_api_key, *args, **kwargs):
        super().__init__(openai_api_key=openai_api_key, *args, **kwargs)
        
    def _embed_documents(self, texts):
        return super().embed_documents(texts)

    def __call__(self, input):
        return self._embed_documents(input)


def get_db(collection_name):
    vector_client = chromadb.HttpClient(host="172.22.0.7", port=8000)

    openai_embedding = CustomOpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))

    vectordb = Chroma(
        client=vector_client,
        collection_name=collection_name,
        embedding_function=openai_embedding,
    )
    
    return vectordb