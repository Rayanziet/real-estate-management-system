#this file is essential to follow the singleton pattern, and create a single initialization for the models
#before defining the file, the models were being defined in all classes, thus when the adk agent was being tested for the first time,
#the mcp server's terminal was downloading the BAAI/bge-m3 multiple times equal to the number of times being defined in the files

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

REAL_ESTATE_API_KEY = os.getenv("REAL_ESTATE_API_KEY")
LLM_API_KEY = os.getenv("LLM_API_KEY")
CHROMA_PATH = "chroma"

_embedder = None
_chroma_db = None
_llm_model = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = HuggingFaceEmbeddings(model="BAAI/bge-m3")
    return _embedder

def get_chroma_db():
    global _chroma_db
    if _chroma_db is None:
        embedder = get_embedder()
        _chroma_db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embedder
        )
    return _chroma_db

def get_llm_model():
    global _llm_model
    if _llm_model is None:
        _llm_model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.3,
            max_retries=5,
            api_key=LLM_API_KEY,
        )
    return _llm_model

