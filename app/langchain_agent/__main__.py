import logging
import os
import sys
import httpx
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryPushNotifier, InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from property_search import RealEstateAgent
from agent_executor import RealEstateAgentExecutor
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MissingAPIKeyError(Exception):
    """Exception for missing API key."""


def main():
    """Starts the Real Estate Agent A2A server."""
    host = "localhost"
    port = 10005  
    
    try:
        if not os.getenv("LLM_API_KEY"):
            raise MissingAPIKeyError("LLM_API_KEY environment variable not set.")
        
        if not os.getenv("REAL_ESTATE_API_KEY"):
            raise MissingAPIKeyError("REAL_ESTATE_API_KEY environment variable not set.")
        
        capabilities = AgentCapabilities(streaming=True, pushNotifications=True)
        
        skills = [
            AgentSkill(
                id="property_search",
                name="Property Search Tool",
                description="Search for real estate properties based on criteria like location, price, size, and type",
                tags=["real_estate", "property", "search", "housing"],
                examples=[
                    "Find 3-bedroom houses under $400k in Austin, TX",
                    "Show me condos in downtown Seattle with 2+ bathrooms",
                    "Search for properties with at least 1500 sq ft in ZIP code 90210"
                ],
            ),
            AgentSkill(
                id="parameter_extraction",
                name="Query Parameter Extraction",
                description="Extract structured search parameters from natural language property queries",
                tags=["nlp", "extraction", "parameters"],
                examples=[
                    "I want a house in Miami with 4 bedrooms under 500k",
                    "Looking for a 2-bed apartment in Brooklyn, preferably under $3000/month"
                ],
            )
        ]
        
        agent_card = AgentCard(
            name="Real Estate Search Agent",
            description="Specialized agent for searching real estate properties. Can extract search criteria from natural language queries and return detailed property listings with photos, prices, and contact information.",
            url=f"http://{host}:{port}/",
            version="1.0.0",
            defaultInputModes=RealEstateAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=RealEstateAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=skills,
        )
        
        httpx_client = httpx.AsyncClient()
        request_handler = DefaultRequestHandler(
            agent_executor=RealEstateAgentExecutor(),
            task_store=InMemoryTaskStore(),
            push_notifier=InMemoryPushNotifier(httpx_client),
        )
        
        server = A2AStarletteApplication(
            agent_card=agent_card, 
            http_handler=request_handler
        )
        
        logger.info(f"Starting Real Estate Agent A2A server on {host}:{port}")
        logger.info(f"Agent capabilities: {[skill.name for skill in skills]}")
        
        uvicorn.run(server.build(), host=host, port=port)
        
    except MissingAPIKeyError as e:
        logger.error(f"Configuration Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred during server startup: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()