from mcp.server.fastmcp import FastMCP
import ast
import os
import warnings
import json
import pandas as pd
import ast
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from helper_function import load_instruction_from_file, rag_pipeline, gmail_create_draft, get_distance_and_time, search_nearby
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)


load_dotenv()

REAL_ESTATE_API_KEY = os.getenv("REAL_ESTATE_API_KEY")
LLM_API_KEY = os.getenv("LLM_API_KEY")


model = ChatGoogleGenerativeAI(
    model = "gemini-2.0-flash",
    temperature = 0.3,
    max_retries = 5,
    api_key = LLM_API_KEY,
)

mcp = FastMCP(
    name="Real-Estate-adk-MCP",
    host='127.0.0.1',
    port=8001
)

@mcp.tool()
def rag_pipeline_tool(query: str) -> str:
    """
    Retrieve relevant context from real estate documents using RAG (Retrieval-Augmented Generation).
    
    Args:
        query: User's question or request for information
    
    Returns:
        Answer to the question based on retrieved context
    """
    prompt = rag_pipeline(query)
    try:
        response = model.invoke(prompt)
        if hasattr(response, "content"):
            return response.content
        elif isinstance(response, dict) and "content" in response:
            return response["content"]
        else:
            return str(response)
    except Exception as e:
        return f"Error: {e}"

@mcp.tool()
def gmail_create_draft_tool(recipient: str, agent_name: str, address: str) -> str:
    """
    Create a draft email in Gmail.
    """
    draft = gmail_create_draft(recipient, agent_name, address)
    if draft:
        return f"Draft created successfully: {draft['id']}"
    return "Failed to create draft."

@mcp.tool()
def get_distance(origin: str, destination: str, mode: str) -> dict:
    """
    Get distance and travel time between two locations using Google Maps API.
    
    Args:
        origin: Starting location (address or lat/lng)
        destination: Destination location (address or lat/lng)
        mode: Travel mode (driving, walking, bicycling, transit)
    
    Returns:
        Dictionary with distance and duration or error message.
    """
    distance, duration = get_distance_and_time(origin, destination, mode)
    if distance and duration:
        return {"distance": distance, "duration": duration}
    else:
        return {"error": "Unable to get distance and duration"}
    
@mcp.tool()
def search_nearby_places(address: str, place_type: str) -> list:
    """
    Search for nearby places of a specific type within a certain radius.

    Args:
        address: The address to search from.
        place_type: The type of place to search for (e.g., 'school', 'hospital').

    Returns:
        A list of nearby places matching the criteria.
    """
    try:
        results = search_nearby(address, place_type)
        return results
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    mcp.run(transport="sse")