
from langchain_huggingface import HuggingFaceEmbeddings
import requests
import os
from dotenv import load_dotenv
from rag_pipeline_csv import csv_pipeline
from rag_pipeline_pdf import pdf_pipeline
from langchain.prompts import ChatPromptTemplate
from config import get_chroma_db, get_embedder, get_llm_model
import base64
import os.path
from google.auth.transport.requests import Request
from email.message import EmailMessage
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


load_dotenv()

REAL_ESTATE_API_KEY = os.getenv("REAL_ESTATE_API_KEY")
LLM_API_KEY = os.getenv("LLM_API_KEY")
GOOGLE_MAP_API_KEY = os.getenv("GOOGLE_Map_API_KEY")
SCOPES = ['https://www.googleapis.com/auth/gmail.compose'] #this is to allow the app to create/modify draft emails

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
    #db.count() but it doesn't work, and i coudn't find a proper condition on google

    results = db.similarity_search_with_score(query, k=5)
    #similarity_search_with_score will manage embedding the query and searching for it

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query)

    return prompt


# Ref for the email functionality:
# https://www.youtube.com/watch?v=N3vHJcHBS-w&list=WL&index=1&t=511s
def email_msg(agent_name: str, property_address: str):
    """
    Generates a professional email template for requesting a meeting about a property.
    
    Args:
        agent_name (str): Name of the real estate agent
        property_address (str): Address of the property
        your_name (str): Your full name
        your_contact (str): Your contact info (phone/email)
    
    Returns:
        subject (str), body (str)
    """
    subject = f"Inquiry about property listed on RentCast"

    body = f"""
Dear {agent_name},

I hope this message finds you well.

I am reaching out regarding the property listed on RentCast at {property_address}. I am very interested and would like to schedule a meeting at your earliest convenience to discuss further details.

Please let me know a suitable time for us to connect. You can reach me via this email.

Looking forward to your response.

Best regards,
Rayan Zaitouni
"""
    return subject, body


def get_gmail_service():
    """Gets valid user credentials from storage and creates Gmail API service.
    
    Returns:
        Service object for Gmail API calls
    """
    creds = None
    token_path = os.path.expanduser(os.getenv('GOOGLE_TOKEN_PATH'))
    credentials_path = os.path.expanduser(os.getenv('GOOGLE_CREDENTIALS_PATH'))
    
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(credentials_path):
                raise FileNotFoundError(f"Credentials file not found at {credentials_path}")
            
            flow = InstalledAppFlow.from_client_secrets_file(
                credentials_path, SCOPES)
            creds = flow.run_local_server(port=0)
            
        os.makedirs(os.path.dirname(token_path), exist_ok=True)
        with open(token_path, 'w') as token:
            token.write(creds.to_json())
    
    return build('gmail', 'v1', credentials=creds)

def gmail_create_draft(recipient: str, agent_name: str, address: str):
    """
    Create and insert a draft email in the user's Gmail account.

    This function uses the Gmail API to generate a draft email with a
    professionally formatted message intended for contacting a real estate
    agent about a specific property.

    Args:
        recipient (str): The recipient's email address (agent's email).
        agent_name (str): The real estate agent's name.
        address (str): The property address to reference in the email.

    Returns:
        dict | None: A dictionary containing the created draft's metadata
        (including draft ID and message details) if successful,
        or None if an error occurred.
    """
    try:
        service = get_gmail_service()

        message = EmailMessage()
        subject, body = email_msg(agent_name, address)
        message.set_content(body)

        message["To"] = recipient
        message["From"] = os.getenv("USER_EMAIL")
        message["Subject"] = subject

        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

        create_message = {"message": {"raw": encoded_message}}
        draft = (
            service.users()
            .drafts()
            .create(userId="me", body=create_message)
            .execute()
        )

    except HttpError as error:
        print(f"An error occurred: {error}")
        draft = None

    return draft

# Ref: https://www.youtube.com/watch?v=s8XzpiWfq9I
def get_distance_and_time(origin, destination, mode):
    url = "https://maps.googleapis.com/maps/api/distancematrix/json"

    params = {
            "origins": origin,
            "destinations": destination,
            "mode": mode,
            "units": "metric",
            "key": GOOGLE_MAP_API_KEY
    }

    response = requests.get(url, params=params)
    data = response.json()

    if data["status"] == "OK":
        element = data["rows"][0]["elements"][0]
        if element["status"] == "OK":
            distance = element["distance"]["text"]
            duration = element["duration"]["text"]
            return distance, duration
        else:
            return None, f"Error: {element['status']}"
    else:
        return None, f"API Error: {data['status']}"
    

def get_coordinates(address):
    """Convert address to latitude/longitude using Geocoding API"""
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": address, "key": GOOGLE_MAP_API_KEY}
    response = requests.get(url, params=params).json()

    if response["status"] == "OK":
        location = response["results"][0]["geometry"]["location"]
        return location["lat"], location["lng"]
    else:
        raise Exception(f"Geocoding error: {response['status']}")

def search_nearby(address, place_type):
    lat, lng = get_coordinates(address)
    radius = 2000
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{lat},{lng}",
        "radius": radius,
        "type": place_type,
        "key": GOOGLE_MAP_API_KEY
    }
    response = requests.get(url, params=params).json()

    if response["status"] == "OK":
        return [place["name"] for place in response["results"]]
    else:
        raise Exception(f"Places API error: {response['status']}")
    