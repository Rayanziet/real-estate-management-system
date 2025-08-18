import json
import urllib.parse
from langchain_huggingface import HuggingFaceEmbeddings
import requests
import os
from dotenv import load_dotenv
from rag_pipeline_csv import csv_pipeline
from rag_pipeline_pdf import pdf_pipeline
from langchain.prompts import ChatPromptTemplate
from config import get_chroma_db, get_embedder, get_llm_model


load_dotenv()

REAL_ESTATE_API_KEY = os.getenv("REAL_ESTATE_API_KEY")
LLM_API_KEY = os.getenv("LLM_API_KEY")

def load_instruction_from_file(
    filename: str, default_instruction: str = "Default instruction."
) -> str:
    """Reads instruction text from a file relative to this script."""
    instruction = default_instruction
    try:
        filepath = os.path.join(os.path.dirname(__file__), filename)
        with open(filepath, "r", encoding="utf-8") as f:
            instruction = f.read()
        print(f"Successfully loaded instruction from {filename}")
    except FileNotFoundError:
        print(f"WARNING: Instruction file not found: {filepath}. Using default.")
    except Exception as e:
        print(f"ERROR loading instruction file {filepath}: {e}. Using default.")
    return instruction



def helper_search(input: dict) -> dict:
    url = 'https://api.rentcast.io/v1/listings/sale'
    params = {key:value for key, value in input.items() if value is not None}
    headers = {"accept": "application/json",
               "X-API-Key": REAL_ESTATE_API_KEY}
    response = requests.get(url, headers=headers, params=params)
    result = response.json()
    return result



CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
You are a professional real estate investment advisor. 
Always give clear, confident, and actionable recommendations without disclaimers. 
Do not mention uncertainty, further research, or data limitations. 
Stay strictly focused on the exact location asked about (state or city). 
Ignore similarly named places in other states unless the user explicitly asks about them.
Include relevant numbers (e.g., property tax rates, median home values) wherever possible to support your recommendation.

Context:
{context}

Question: {question}

Answer in the following structured format:
1. Start with a clear yes/no or strong statement about whether the location is good for investment regarding the question asked.
2. List the **best areas** (counties/cities) in that location with reasons why they are attractive, including relevant numbers like tax rates or property values.
3. List the **worst areas** (counties/cities) in that location with reasons why they are less attractive, including relevant numbers if available.
4. End with a **bold, concise summary** that wraps up the recommendation.

Use bullet points for best/worst areas, and keep the tone professional, decisive, and actionable.
"""

#Ref for this function: https://github.com/pixegami/rag-tutorial-v2/blob/main/query_data.py
def rag_pipeline(query: str):
    db = get_chroma_db()

    items = db.get(include=[]) 
    if len(items["ids"]) == 0:
        pdf_pipeline()
        csv_pipeline()
    #this condition was referenced by chatgpt, only to check if db is empty so we run the pipelines
    #I tried db.count() but it didnt work, and i coudn't find a proper condition on google

    results = db.similarity_search_with_score(query, k=5)
    #similarity_search_with_score will manage embedding the query and searching for it

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query)

    return prompt
