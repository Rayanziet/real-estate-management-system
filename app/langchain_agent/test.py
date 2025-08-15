from helper_function import helper_search, load_instruction_from_file
from property_search import model
import ast
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
    return ast.literal_eval(response.content)

query = "Find 3 bedroom Single Family houses in 123 Main Street, Los Angeles, CA 91234, with a price between $150k and $300k, and 2 bathrooms, get only 5"
params = extract_param(query)
print(params)

# output : 
# Successfully loaded instruction from extraction_instructions.txt
# {'address': '123 Main Street, Los Angeles, CA, 91234', 'bedrooms': 3, 'bathrooms': 2, 'propertyType': 'Single Family', 'priceMin': 150000, 'priceMax': 300000, 'limit': 5}