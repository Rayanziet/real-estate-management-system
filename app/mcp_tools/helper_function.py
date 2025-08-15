import json
import urllib.parse
import requests
import os
from dotenv import load_dotenv

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
