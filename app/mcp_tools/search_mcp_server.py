from mcp.server.fastmcp import FastMCP
from helper_function import helper_search, load_instruction_from_file
import ast
import os
import warnings
import json
import pandas as pd
import ast
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from helper_function import helper_search, load_instruction_from_file
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
    name="Real-Estate-MCP",
    host='0.0.0.0',
    port='8787'
)

@mcp.tool()
def extract_param( query : str) -> dict:
    """
    Extract structured search parameters from natural language property query.
    
    Args:
        query: Natural language query about property search
        
    Returns:
        JSON string of extracted search parameters
    """
    instructions = load_instruction_from_file("extraction_instructions.txt")
    response = model.invoke(f"Parse this query {query} into structured parameters' dictionary using these instructions: {instructions}")
    #Although we mentioned to parse into a dictionary, this response might not be in the expected format and might be a string
    #https://stackoverflow.com/questions/39807724/extract-python-dictionary-from-string
    return ast.literal_eval(response.content)

@mcp.tool()
def search_properties(input: dict):
    """
    Search for real estate sale listings based on location, price, and property features.
    
    Args:
        input: Search parameters including city, state, zipCode, priceMin, priceMax, 
                 bedroomsMin, bathroomsMax, squareFeetMin, propertyType, limit, etc.
    
    Returns:
        Formatted response about properties matching the search criteria.
    """
    result = helper_search(input)
    instructions = load_instruction_from_file("agent_response_instructions.txt")
    response = model.invoke(f"You are a professional and friendly real estate agent.  The client's search parameters are: {json.dumps(input, indent=2)} Here are the properties found: {json.dumps(result, indent=2)} Follow these instructions when preparing your response: {instructions}")
    return response.content

if __name__ == "__main__":
    mcp.run(transport="sse")