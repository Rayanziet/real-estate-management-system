import uvicorn
import json
import asyncio
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from property_search import real_estate_agent


class RealEstateAgentExecutor:
    """Simple agent executor that delegates to LangChain agent."""
    
    async def execute(self, request, queue):
        """Execute method for non-streaming requests."""
        try:
            print("DEBUG: === EXECUTE METHOD START ===")
            
            # Extract the query from the A2A request
            query = self._extract_query_from_request(request)
            print(f"DEBUG: Query received: '{query}'")
            
            if not query:
                return "❌ No search query provided. Please specify what you're looking for."
            
            # Simply call the LangChain agent - it will handle parameter extraction and search
            print("DEBUG: Calling LangChain real_estate_agent...")
            response_text, state = await real_estate_agent(query)
            
            print(f"DEBUG: LangChain agent returned response (length: {len(response_text)})")
            
            # Return the formatted response directly
            final_response = str(response_text).strip() if response_text else "❌ No properties found"
            print(f"DEBUG: Returning response: {final_response[:100]}...")
            
            return final_response
            
        except Exception as e:
            error_message = f"Agent execution error: {str(e)}"
            print(f"DEBUG: Exception in execute: {error_message}")
            import traceback
            traceback.print_exc()
            
            # Return user-friendly error messages
            if "Connection refused" in str(e) or "Failed to connect" in str(e):
                return "❌ Unable to access property database. Please ensure the MCP server is running on port 8787."
            elif "timeout" in str(e).lower():
                return "⏰ Property search timed out. Please try again or refine your search criteria."
            else:
                return f"❌ Property search error: {str(e)}"
        finally:
            print("DEBUG: === EXECUTE METHOD END ===")
    
    async def execute_streaming(self, request, queue):
        """Execute method for streaming responses."""
        try:
            print("DEBUG: === STREAMING EXECUTE METHOD START ===")
            
            # Extract the query
            query = self._extract_query_from_request(request)
            print(f"DEBUG: Streaming query: {query}")
            
            if not query:
                if queue and not queue.is_closed():
                    await queue.enqueue_event("❌ No search query provided.")
                return "No query provided"
            
            # Send initial status
            if queue and not queue.is_closed():
                await queue.enqueue_event("🔍 Starting property search...")
            
            # Call the LangChain agent
            print("DEBUG: Calling LangChain real_estate_agent for streaming...")
            response_text, state = await real_estate_agent(query)
            
            print(f"DEBUG: Streaming response length: {len(response_text)}")
            
            # Send the formatted response
            if queue and not queue.is_closed():
                await queue.enqueue_event(f"✅ Property search results:\n{response_text}")
                
                # Signal completion
                await queue.enqueue_event("🏁 Search completed.")
            
            return "Streaming property search completed"
            
        except Exception as e:
            error_message = f"Streaming execution error: {str(e)}"
            print(f"DEBUG: Exception in streaming: {error_message}")
            
            if queue and not queue.is_closed():
                try:
                    error_msg = "❌ Unable to access property database. Please ensure the MCP server is running." if "Connection refused" in str(e) else f"❌ {error_message}"
                    await queue.enqueue_event(error_msg)
                except:
                    pass
            
            return error_message
        finally:
            print("DEBUG: === STREAMING EXECUTE METHOD END ===")
    
    def _extract_query_from_request(self, request):
        """Extract the query text from A2A request."""
        try:
            # Method 1: Direct message attribute
            if hasattr(request, 'message') and hasattr(request.message, 'parts'):
                text_parts = []
                for part in request.message.parts:
                    if hasattr(part, 'root') and hasattr(part.root, 'text'):
                        text_parts.append(str(part.root.text))
                    elif hasattr(part, 'text'):
                        text_parts.append(str(part.text))
                
                query = ' '.join(text_parts)
                if query:
                    return query.strip()
            
            # Method 2: Try params.message path
            if hasattr(request, '_params') and hasattr(request._params, 'message'):
                message = request._params.message
                if hasattr(message, 'parts'):
                    text_parts = []
                    for part in message.parts:
                        if hasattr(part, 'root') and hasattr(part.root, 'text'):
                            text_parts.append(str(part.root.text))
                        elif hasattr(part, 'text'):
                            text_parts.append(str(part.text))
                    
                    query = ' '.join(text_parts)
                    if query:
                        return query.strip()
            
            return ""
            
        except Exception as e:
            print(f"DEBUG: Error extracting query: {e}")
            return ""


if __name__ == '__main__':
    # Check A2A version for debugging
    try:
        import a2a
        print(f"DEBUG: A2A version: {a2a.__version__}")
    except:
        print("DEBUG: Could not determine A2A version")
    
    # Define the real estate search skill
    real_estate_skill = AgentSkill(
        id='real_estate_search',
        name='Real Estate Property Search',
        description='Searches for real estate properties based on location, price range, property type, and other specifications.',
        tags=['real-estate', 'property', 'search', 'listings', 'market-analysis'],
        examples=[
            'Find 2-bedroom apartments in Manhattan under $3000',
            'Search for houses with garage in Brooklyn',
            'Properties for sale in Queens with price under $500k',
            'Show me luxury condos in downtown with parking',
            'Find family homes near schools in Staten Island'
        ],
    )

    # Create the agent card
    public_agent_card = AgentCard(
        name='Real Estate Search Agent',
        description='AI-powered real estate agent specializing in property search and market analysis.',
        url='http://localhost:8001/',
        version='1.0.1',
        default_input_modes=['text'],
        default_output_modes=['text', 'json'],
        capabilities=AgentCapabilities(
            streaming=True,
            supports_files=False,
            supports_vision=False
        ),
        skills=[real_estate_skill],
        supports_authenticated_extended_card=False,
        protocol_version='0.3.0',
        preferred_transport='JSONRPC'
    )

    # Create the request handler
    try:
        request_handler = DefaultRequestHandler(
            agent_executor=RealEstateAgentExecutor(),
            task_store=InMemoryTaskStore(),
        )

        # Create the A2A server application
        server = A2AStarletteApplication(
            agent_card=public_agent_card,
            http_handler=request_handler,
        )

        print("🚀 Starting Real Estate A2A Agent Server...")
        print("📍 Server will be available at: http://localhost:8001")
        print("🔗 Agent card endpoint: http://localhost:8001/.well-known/agent")
        print("📡 Ready to receive A2A requests from orchestrator...")

        # Run the server
        uvicorn.run(
            server.build(), 
            host='0.0.0.0', 
            port=8001,
            log_level="info"
        )
        
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        import traceback
        traceback.print_exc()
