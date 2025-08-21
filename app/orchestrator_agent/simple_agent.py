import logging
import asyncio
from typing import Dict, Any, Optional
from enum import Enum

# Import existing orchestrators
from orch_agent import Orchestrator as LangChainOrchestrator
from adk_orchestrator import ADKOrchestrator


class OrchestratorType(Enum):
    """Enum for orchestrator types."""
    LANGCHAIN = "langchain"
    ADK = "adk"


class SimpleAgent:
    """
    Simple agent that routes queries to the appropriate orchestrator
    and formats responses according to instructions.
    """
    
    def __init__(self, langchain_url: str, adk_url: str):
        self.langchain_url = langchain_url
        self.adk_url = adk_url
        
        # Initialize orchestrators
        self.langchain_orchestrator = None
        self.adk_orchestrator = None
        
        # Load instructions
        self.instructions = self._load_instructions()
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("🚀 Simple Agent initialized")
    
    def _load_instructions(self) -> str:
        """Load instructions from the instructions file."""
        try:
            with open('instructions.txt', 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            # Default instructions if file not found
            return """
            RESPONSE FORMATTING INSTRUCTIONS:
            
            1. Always provide clear, professional responses
            2. Structure information in a logical way
            3. Use bullet points for lists
            4. Be concise but informative
            5. Maintain a helpful and professional tone
            6. If technical details are provided, explain them simply
            7. Always acknowledge the source (which orchestrator was used)
            """
    
    async def initialize(self):
        """Initialize both orchestrators."""
        try:
            # Initialize LangChain orchestrator
            self.logger.info("🔄 Initializing LangChain orchestrator...")
            self.langchain_orchestrator = LangChainOrchestrator({
                "real_estate": self.langchain_url
            })
            await self.langchain_orchestrator.initialize_clients()
            self.logger.info("✅ LangChain orchestrator initialized")
            
            # Initialize ADK orchestrator
            self.logger.info("🔄 Initializing ADK orchestrator...")
            self.adk_orchestrator = ADKOrchestrator(self.adk_url)
            await self.adk_orchestrator.initialize_clients()
            self.logger.info("✅ ADK orchestrator initialized")
            
            self.logger.info("🎯 All orchestrators initialized successfully!")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize orchestrators: {e}")
            raise
    
    def _route_query(self, query: str) -> OrchestratorType:
        """
        Simple routing logic based on keywords.
        Returns which orchestrator should handle the query.
        """
        query_lower = query.lower()
        
        # LangChain keywords (property search, database queries, market data)
        langchain_keywords = [
            "search", "find", "properties", "property", "house", "apartment", "listing", 
            "available", "for sale", "for rent", "database", "query", "show me",
            "market trends", "market data", "trends", "statistics", "prices", "comparison",
            "zip code", "under $", "over $", "bedroom", "bathroom", "sq ft", "square feet"
        ]
        
        # ADK keywords (document analysis, communication, location, guidance)
        adk_keywords = [
            "analyze", "document", "tax", "report", "pdf", "csv", "data analysis",
            "email", "draft", "send", "communication", "client", "inquiry", "create",
            "nearby", "schools", "restaurants", "grocery", "amenities", "neighborhood", 
            "distance", "travel", "commute", "how long", "calculate", "route", 
            "driving", "transit", "guidance", "advice", "consider", "factors", 
            "tips", "help", "insights", "recommendations", "what are", "explain"
        ]
        
        # Count matches with weighted scoring
        langchain_score = 0
        adk_score = 0
        
        # Check for LangChain patterns
        for keyword in langchain_keywords:
            if keyword in query_lower:
                # Give higher weight to specific property search terms
                if keyword in ["properties", "property", "house", "apartment", "listing", "for sale", "for rent"]:
                    langchain_score += 2
                else:
                    langchain_score += 1
        
        # Check for ADK patterns
        for keyword in adk_keywords:
            if keyword in query_lower:
                # Give higher weight to specific analysis/guidance terms
                if keyword in ["analyze", "document", "email", "draft", "nearby", "calculate", "guidance", "advice"]:
                    adk_score += 2
                else:
                    adk_score += 1
        
        # Route based on weighted scores
        if langchain_score > adk_score:
            return OrchestratorType.LANGCHAIN
        elif adk_score > langchain_score:
            return OrchestratorType.ADK
        else:
            # Default to ADK for more comprehensive handling
            return OrchestratorType.ADK
    
    def _extract_content_from_response(self, response: Dict[str, Any], orchestrator_type: OrchestratorType) -> str:
        """
        Extract content from the complex A2A response structures.
        Handles both LangChain and ADK response formats.
        """
        try:
            # Check if response has success field
            if 'success' in response and not response['success']:
                return response.get('content', 'Request failed')
            
            # Extract content based on orchestrator type
            if orchestrator_type == OrchestratorType.LANGCHAIN:
                # LangChain response structure
                if 'content' in response:
                    return response['content']
                elif 'response' in response:
                    # Try to extract from nested response
                    nested_response = response['response']
                    if isinstance(nested_response, dict):
                        # Look for content in result structure
                        if 'result' in nested_response:
                            result = nested_response['result']
                            if isinstance(result, dict) and 'content' in result:
                                return str(result['content'])
                        # Fallback to string representation
                        return str(nested_response)
                    return str(nested_response)
                else:
                    return "Response received (content extraction failed)"
            
            else:  # ADK response
                # ADK response structure - look for the actual content
                if 'content' in response:
                    return response['content']
                elif 'chunk_data' in response:
                    chunk_data = response['chunk_data']
                    if isinstance(chunk_data, dict):
                        # Look for content in chunk_data
                        if 'content' in chunk_data:
                            return chunk_data['content']
                        # Check if this is a final chunk with content
                        elif 'final' in chunk_data and chunk_data['final']:
                            if 'content' in chunk_data:
                                return chunk_data['content']
                            else:
                                return "Final response received (content extraction failed)"
                        # Check for text field
                        elif 'text' in chunk_data:
                            return chunk_data['text']
                        # Check for root.text structure
                        elif 'root' in chunk_data and isinstance(chunk_data['root'], dict):
                            if 'text' in chunk_data['root']:
                                return chunk_data['root']['text']
                
                # If we still don't have content, try to extract from the original response structure
                if 'original_response' in response:
                    orig_response = response['original_response']
                    if isinstance(orig_response, dict):
                        # Look for content in various possible locations
                        for key in ['content', 'text', 'message']:
                            if key in orig_response:
                                return str(orig_response[key])
                        
                        # Check for nested structures
                        if 'result' in orig_response:
                            result = orig_response['result']
                            if isinstance(result, dict):
                                for key in ['content', 'text', 'message']:
                                    if key in result:
                                        return str(result[key])
                
                # Final fallback - return a debug message
                return f"Response received but content extraction failed. Response keys: {list(response.keys())}"
            
        except Exception as e:
            self.logger.error(f"Error extracting content: {e}")
            return f"Response received (extraction error: {e})"
    
    def _format_response(self, response: Dict[str, Any], orchestrator_type: OrchestratorType, query: str) -> str:
        """
        Format the response according to instructions.
        Creates a well-structured, professional response.
        """
        try:
            # Extract content using the new method
            content = self._extract_content_from_response(response, orchestrator_type)
            
            # Check if response indicates failure
            success = response.get('success', True)  # Default to True if not present
            
            if not success:
                error = response.get('error', 'Unknown error')
                return f"""
❌ **Query Processing Failed**

**Original Query:** {query}
**Error:** {error}
**Orchestrator Used:** {orchestrator_type.value.title()}

Please try rephrasing your query or contact support if the issue persists.
                """.strip()
            
            # Format successful response
            orchestrator_name = "LangChain Real Estate Agent" if orchestrator_type == OrchestratorType.LANGCHAIN else "ADK AI Agent"
            
            formatted_response = f"""
✅ **Query Processed Successfully**

**Original Query:** {query}
**Orchestrator Used:** {orchestrator_name}

**Response:**
{content}

---
*This response was generated using the {orchestrator_name} based on your query requirements.*
            """.strip()
            
            return formatted_response
            
        except Exception as e:
            self.logger.error(f"Error formatting response: {e}")
            return f"""
❌ **Response Formatting Error**

**Original Query:** {query}
**Error:** {str(e)}
**Orchestrator Used:** {orchestrator_type.value.title()}

Please contact support for assistance.
            """.strip()
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        Main method to process queries through the appropriate orchestrator.
        """
        try:
            self.logger.info(f"🎯 Processing query: {query[:100]}...")
            
            # Route the query
            orchestrator_type = self._route_query(query)
            self.logger.info(f"🚀 Routing to: {orchestrator_type.value}")
            
            # Process with selected orchestrator
            if orchestrator_type == OrchestratorType.LANGCHAIN:
                response = await self._process_with_langchain(query)
            else:
                response = await self._process_with_adk(query)
            
            # Format the response
            formatted_content = self._format_response(response, orchestrator_type, query)
            
            return {
                'success': True,
                'orchestrator_used': orchestrator_type.value,
                'original_response': response,
                'formatted_content': formatted_content,
                'query': query
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error processing query: {e}")
            return {
                'success': False,
                'error': str(e),
                'orchestrator_used': 'none',
                'formatted_content': f"Error processing query: {str(e)}",
                'query': query
            }
    
    async def _process_with_langchain(self, query: str) -> Dict[str, Any]:
        """Process query using LangChain orchestrator."""
        if not self.langchain_orchestrator:
            raise ValueError("LangChain orchestrator not initialized")
        
        try:
            # Use streaming for LangChain to get complete responses
            last_chunk = None
            async for chunk in self.langchain_orchestrator.send_streaming_message("real_estate", query):
                if chunk.get('success', False):
                    # Store the last successful chunk
                    last_chunk = chunk
                    # Check if this is the final chunk with property data
                    chunk_data = chunk.get('chunk_data', {})
                    if chunk_data.get('final', False) or 'final' in chunk_data:
                        return chunk
            
            # Return the last chunk if no final chunk found
            return last_chunk if last_chunk else {
                'success': False,
                'error': 'No response received from LangChain agent',
                'content': 'No response received'
            }
            
        except Exception as e:
            self.logger.error(f"❌ LangChain processing failed: {e}")
            raise
    
    async def _process_with_adk(self, query: str) -> Dict[str, Any]:
        """Process query using ADK orchestrator."""
        if not self.adk_orchestrator:
            raise ValueError("ADK orchestrator not initialized")
        
        try:
            # For ADK, we'll use the streaming method to get the final response
            async for chunk in self.adk_orchestrator.send_streaming_message(query):
                # Get the last chunk which should contain the final response
                if chunk.get('success', False):
                    # Check if this is the final chunk
                    chunk_data = chunk.get('chunk_data', {})
                    if chunk_data.get('final', False) or 'final' in chunk_data:
                        return chunk
                    # Store the last successful chunk
                    last_chunk = chunk
            
            # Return the last chunk if no final chunk found
            return last_chunk if 'last_chunk' in locals() else {
                'success': False,
                'error': 'No response received from ADK agent',
                'content': 'No response received'
            }
            
        except Exception as e:
            self.logger.error(f"❌ ADK processing failed: {e}")
            raise
    
    async def get_status(self) -> Dict[str, Any]:
        """Get status of the agent and orchestrators."""
        status = {
            'agent_status': 'active',
            'langchain_status': 'unknown',
            'adk_status': 'unknown',
            'timestamp': asyncio.get_event_loop().time()
        }
        
        # Check LangChain status
        try:
            if self.langchain_orchestrator:
                # Try to get agent info instead of sending a message
                agents = self.langchain_orchestrator.list_agents()
                if agents and 'real_estate' in agents:
                    status['langchain_status'] = 'active'
                else:
                    status['langchain_status'] = 'error: no agents found'
            else:
                status['langchain_status'] = 'not_initialized'
        except Exception as e:
            status['langchain_status'] = f'error: {str(e)}'
        
        # Check ADK status
        try:
            if self.adk_orchestrator:
                info = self.adk_orchestrator.get_root_agent_info()
                if 'error' not in info:
                    status['adk_status'] = 'active'
                else:
                    status['adk_status'] = 'error: agent info unavailable'
            else:
                status['adk_status'] = 'not_initialized'
        except Exception as e:
            status['adk_status'] = f'error: {str(e)}'
        
        return status
    
    async def cleanup(self):
        """Clean up all orchestrators."""
        try:
            if self.langchain_orchestrator:
                await self.langchain_orchestrator.cleanup()
            if self.adk_orchestrator:
                await self.adk_orchestrator.cleanup()
            self.logger.info("🧹 All orchestrators cleaned up")
        except Exception as e:
            self.logger.error(f"❌ Error during cleanup: {e}")


# ==== Test Function ====
async def main():
    """Test the simple agent."""
    
    # Configure endpoints
    langchain_url = "http://localhost:10005"  # Your existing LangChain agent
    adk_url = "http://localhost:10002"        # Your ADK agent
    
    agent = SimpleAgent(langchain_url, adk_url)
    
    try:
        print("🚀 Starting Simple Agent Test")
        print("=" * 50)
        
        # Initialize
        await agent.initialize()
        print("✅ All orchestrators initialized!")
        
        # Test different query types with REAL US addresses
        test_queries = [
            # Should route to LangChain (property search/database)
            "Find properties for sale in downtown Austin, TX under $500,000",  # LangChain
            
            # Should route to ADK (document analysis)
            "Analyze this property tax document: The property at 1234 Congress Ave, Austin, TX 78701 has an assessed value of $450,000 with annual property taxes of $5,400. The tax rate is 1.2% and includes school district, county, and city levies.",  # ADK
            
            # Should route to ADK (email communication)
            "Create a professional email draft for a property inquiry. Send to: testing@example.com. Subject: Property Inquiry for 1234 Congress Ave, Austin, TX. Message: I'm interested in learning more about this property. Can you provide details about recent sales in the area and any upcoming open houses?",  # ADK
            
            # Should route to ADK (location amenities)
            "Find nearby amenities around 1234 Congress Ave, Austin, TX 78701. I need information about schools, restaurants, grocery stores, and public transportation within a 1-mile radius.",  # ADK
            
            
            # Should route to ADK (distance calculation)
            "Calculate travel time and distance from 1234 Congress Ave, Austin, TX to Austin-Bergstrom International Airport, and also to Dell Seton Medical Center. I need both driving and public transit options.",  # ADK
        
            
            # Should route to ADK (general guidance)
            "What are the key factors I should consider when buying a first home in Miami Beach, FL? Include information about market trends, property taxes, and local regulations.",  # ADK
        ]
        
        print("\n🧪 Testing Query Routing and Response Formatting:")
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Test {i} ---")
            print(f"Query: {query}")
            
            try:
                response = await agent.process_query(query)
                print(f"🤖 Orchestrator: {response['orchestrator_used']}")
                print(f"📝 Response:")
                print(f"{response['formatted_content']}")
                print("-" * 80)
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("\n✅ Test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'agent' in locals():
            await agent.cleanup()
            print("🧹 Cleanup completed")


if __name__ == "__main__":
    asyncio.run(main())
