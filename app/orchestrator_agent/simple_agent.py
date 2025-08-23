import logging
import asyncio
from typing import Dict, Any, Optional, List
from enum import Enum

from orch_agent import Orchestrator as LangChainOrchestrator
from adk_orchestrator import ADKOrchestrator

from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

class OrchestratorType(Enum):
    """Enum for orchestrator types."""
    LANGCHAIN = "langchain"
    ADK = "adk"


class SimpleAgent:
    """
    Enhanced simple agent that manages conversations, routes queries to orchestrators,
    and formats responses using Gemini 2.0 for professional output.
    """
    
    def __init__(self, langchain_url: str, adk_url: str):
        self.langchain_url = langchain_url
        self.adk_url = adk_url
        
        self.langchain_orchestrator = None
        self.adk_orchestrator = None
        
        self.gemini_model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.3,
            max_retries=5,
            api_key=os.getenv("LLM_API_KEY")
        )
        
        self.conversation_history: List[Dict[str, str]] = []
        
        self.instructions = self._load_instructions()
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    
    def _load_instructions(self) -> str:
        """Load instructions from the instructions file."""
        try:
            with open('instructions.txt', 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return """
            RESPONSE FORMATTING INSTRUCTIONS:
            
            1. Always provide clear, professional responses
            2. Structure information in a logical way
            3. Use bullet points for lists
            4. Be concise but informative
            5. Maintain a helpful and professional tone
            6. If technical details are provided, explain them simply
            7. Always acknowledge the source (which orchestrator was used)
            8. Maintain conversation context and flow
            """
    
    def _add_to_conversation_history(self, role: str, content: str):
        """Add message to conversation history."""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": asyncio.get_event_loop().time()
        })
        
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def _get_conversation_context(self) -> str:
        """Get recent conversation context for Gemini."""
        if not self.conversation_history:
            return ""
        
        context = "Recent conversation context:\n"
        for msg in self.conversation_history[-5:]:  # Last 5 messages
            context += f"{msg['role'].title()}: {msg['content']}\n"
        return context
    
    async def _format_response_with_gemini(self, raw_response: str, user_query: str, orchestrator_used: str) -> str:
        """Use Gemini 2.0 to format and enhance the response."""
        try:
            conversation_context = self._get_conversation_context()
            
            formatting_prompt = f"""
            You are a professional real estate assistant. Format the following response to be user-friendly and professional.
            
            {conversation_context}
            
            USER'S ORIGINAL QUERY: {user_query}
            ORCHESTRATOR USED: {orchestrator_used}
            RAW RESPONSE FROM AGENT: {raw_response}
            
            Please format this response to be:
            1. Professional and helpful
            2. Easy to understand
            3. Well-structured with clear sections
            4. Conversational and engaging
            5. Maintains context from the conversation
            6. Adds helpful insights when appropriate
            
            Format the response professionally and make it sound natural in conversation.
            """
            
            response = self.gemini_model.invoke(formatting_prompt)
            formatted_response = response.content if hasattr(response, 'content') else str(response)
            
            return formatted_response
            
        except Exception as e:
            self.logger.error(f"Error formatting response with Gemini: {e}")
            return raw_response
    
    async def initialize(self):
        """Initialize both orchestrators."""
        try:
            self.langchain_orchestrator = LangChainOrchestrator({
                "real_estate": self.langchain_url
            })
            await self.langchain_orchestrator.initialize_clients()

            self.adk_orchestrator = ADKOrchestrator(self.adk_url)
            await self.adk_orchestrator.initialize_clients()

            
        except Exception as e:
            self.logger.error(f"Failed to initialize orchestrators: {e}")
            raise
    
    def _route_query(self, query: str) -> OrchestratorType:
        """
        Simple routing logic based on keywords.
        Returns which orchestrator should handle the query.
        """
        query_lower = query.lower()
        
        langchain_keywords = [
            "search", "find", "properties", "property", "house", "apartment", "listing", 
            "available", "for sale", "for rent", "database", "query", "show me",
            "market trends", "market data", "trends", "statistics", "prices", "comparison",
            "zip code", "under $", "over $", "bedroom", "bathroom", "sq ft", "square feet"
        ]
        
        adk_keywords = [
            "analyze", "document", "tax", "report", "pdf", "csv", "data analysis",
            "email", "draft", "send", "communication", "client", "inquiry", "create",
            "nearby", "schools", "restaurants", "grocery", "amenities", "neighborhood", 
            "distance", "travel", "commute", "how long", "calculate", "route", 
            "driving", "transit", "guidance", "advice", "consider", "factors", 
            "tips", "help", "insights", "recommendations", "what are", "explain"
        ]
        
        langchain_score = 0
        adk_score = 0
        
        for keyword in langchain_keywords:
            if keyword in query_lower:
                if keyword in ["properties", "property", "house", "apartment", "listing", "for sale", "for rent"]:
                    langchain_score += 2
                else:
                    langchain_score += 1
        
        for keyword in adk_keywords:
            if keyword in query_lower:
                if keyword in ["analyze", "document", "email", "draft", "nearby", "calculate", "guidance", "advice"]:
                    adk_score += 2
                else:
                    adk_score += 1
        
        if langchain_score > adk_score:
            return OrchestratorType.LANGCHAIN
        elif adk_score > langchain_score:
            return OrchestratorType.ADK
        else:
            return OrchestratorType.ADK
    
    def _extract_content_from_response(self, response: Dict[str, Any], orchestrator_type: OrchestratorType) -> str:
        """
        Extract content from the complex A2A response structures.
        Handles both LangChain and ADK response formats.
        """
        try:
            if 'success' in response and not response['success']:
                return response.get('content', 'Request failed')
            
            if orchestrator_type == OrchestratorType.LANGCHAIN:
                if 'content' in response:
                    return response['content']
                elif 'response' in response:
                    nested_response = response['response']
                    if isinstance(nested_response, dict):
                        if 'result' in nested_response:
                            result = nested_response['result']
                            if isinstance(result, dict) and 'content' in result:
                                return str(result['content'])
                        return str(nested_response)
                    return str(nested_response)
                else:
                    return "Response received (content extraction failed)"
            
            else: 
                if 'content' in response:
                    return response['content']
                elif 'chunk_data' in response:
                    chunk_data = response['chunk_data']
                    if isinstance(chunk_data, dict):
                        if 'content' in chunk_data:
                            return chunk_data['content']
                        elif 'final' in chunk_data and chunk_data['final']:
                            if 'content' in chunk_data:
                                return chunk_data['content']
                            else:
                                return "Final response received (content extraction failed)"
                        elif 'text' in chunk_data:
                            return chunk_data['text']
                        elif 'root' in chunk_data and isinstance(chunk_data['root'], dict):
                            if 'text' in chunk_data['root']:
                                return chunk_data['root']['text']
                
                if 'original_response' in response:
                    orig_response = response['original_response']
                    if isinstance(orig_response, dict):
                        for key in ['content', 'text', 'message']:
                            if key in orig_response:
                                return str(orig_response[key])
                        
                        if 'result' in orig_response:
                            result = orig_response['result']
                            if isinstance(result, dict):
                                for key in ['content', 'text', 'message']:
                                    if key in result:
                                        return str(result[key])
                
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
            content = self._extract_content_from_response(response, orchestrator_type)
            
            success = response.get('success', True)
            
            if not success:
                error = response.get('error', 'Unknown error')
                return f"""
 **Query Processing Failed**

**Original Query:** {query}
**Error:** {error}
**Orchestrator Used:** {orchestrator_type.value.title()}

Please try rephrasing your query or contact support if the issue persists.
                """.strip()
            
            orchestrator_name = "LangChain Real Estate Agent" if orchestrator_type == OrchestratorType.LANGCHAIN else "ADK AI Agent"
            
            formatted_response = f"""
 **Query Processed Successfully**

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
 **Response Formatting Error**

**Original Query:** {query}
**Error:** {str(e)}
**Orchestrator Used:** {orchestrator_type.value.title()}

Please contact support for assistance.
            """.strip()
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        Main method to process queries through the appropriate orchestrator.
        Enhanced with Gemini 2.0 conversation management.
        """
        try:
            self.logger.info(f"Processing query: {query[:100]}...")
            
            self._add_to_conversation_history("user", query)
            
            orchestrator_type = self._route_query(query)
            self.logger.info(f"Routing to: {orchestrator_type.value}")
            
            if orchestrator_type == OrchestratorType.LANGCHAIN:
                response = await self._process_with_langchain(query)
            else:
                response = await self._process_with_adk(query)
            
            raw_content = self._extract_content_from_response(response, orchestrator_type)
            
            formatted_content = await self._format_response_with_gemini(
                raw_content, 
                query, 
                orchestrator_type.value
            )
            
            self._add_to_conversation_history("assistant", formatted_content)
            
            return {
                'success': True,
                'orchestrator_used': orchestrator_type.value,
                'original_response': response,
                'formatted_content': formatted_content,
                'query': query
            }
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            error_response = f"Error processing query: {str(e)}"
            self._add_to_conversation_history("assistant", error_response)
            
            return {
                'success': False,
                'error': str(e),
                'orchestrator_used': 'none',
                'formatted_content': error_response,
                'query': query
            }
    
    async def _process_with_langchain(self, query: str) -> Dict[str, Any]:
        """Process query using LangChain orchestrator."""
        if not self.langchain_orchestrator:
            raise ValueError("LangChain orchestrator not initialized")
        
        try:
            last_chunk = None
            async for chunk in self.langchain_orchestrator.send_streaming_message("real_estate", query):
                if chunk.get('success', False):
                    last_chunk = chunk
                    chunk_data = chunk.get('chunk_data', {})
                    if chunk_data.get('final', False) or 'final' in chunk_data:
                        return chunk
            
            return last_chunk if last_chunk else {
                'success': False,
                'error': 'No response received from LangChain agent',
                'content': 'No response received'
            }
            
        except Exception as e:
            self.logger.error(f"LangChain processing failed: {e}")
            raise
    
    async def _process_with_adk(self, query: str) -> Dict[str, Any]:
        """Process query using ADK orchestrator."""
        if not self.adk_orchestrator:
            raise ValueError("ADK orchestrator not initialized")
        
        try:
            async for chunk in self.adk_orchestrator.send_streaming_message(query):
                if chunk.get('success', False):
                    chunk_data = chunk.get('chunk_data', {})
                    if chunk_data.get('final', False) or 'final' in chunk_data:
                        return chunk
                    last_chunk = chunk
            
            return last_chunk if 'last_chunk' in locals() else {
                'success': False,
                'error': 'No response received from ADK agent',
                'content': 'No response received'
            }
            
        except Exception as e:
            self.logger.error(f"ADK processing failed: {e}")
            raise
    
    async def get_status(self) -> Dict[str, Any]:
        """Get status of the agent and orchestrators."""
        status = {
            'agent_status': 'active',
            'langchain_status': 'unknown',
            'adk_status': 'unknown',
            'conversation_history_length': len(self.conversation_history)
        }
        
        try:
            if self.langchain_orchestrator:
                agents = self.langchain_orchestrator.list_agents()
                if agents and 'real_estate' in agents:
                    status['langchain_status'] = 'active'
                else:
                    status['langchain_status'] = 'error: no agents found'
            else:
                status['langchain_status'] = 'not_initialized'
        except Exception as e:
            status['langchain_status'] = f'error: {str(e)}'
        

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
    
    async def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the conversation history."""
        return self.conversation_history.copy()
    
    async def clear_conversation_history(self) -> bool:
        """Clear the conversation history."""
        try:
            self.conversation_history.clear()
            return True
        except Exception as e:
            self.logger.error(f"Error clearing conversation history: {e}")
            return False
    
    async def cleanup(self):
        """Clean up all orchestrators."""
        try:
            if self.langchain_orchestrator:
                await self.langchain_orchestrator.cleanup()
            if self.adk_orchestrator:
                await self.adk_orchestrator.cleanup()
            self.logger.info("All orchestrators cleaned up")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


#After testing each orchestrator, I used this main method to test if both will work together properly
async def main():
    """Test the simple agent."""
    
    langchain_url = "http://localhost:10005" 
    adk_url = "http://localhost:10002"        
    
    agent = SimpleAgent(langchain_url, adk_url)
    
    try:

        await agent.initialize()

        test_queries = [
            "Find properties for sale in downtown Austin, TX under $500,000",  # LangChain
            
            "Analyze this property tax document: The property at 1234 Congress Ave, Austin, TX 78701 has an assessed value of $450,000 with annual property taxes of $5,400. The tax rate is 1.2% and includes school district, county, and city levies.",  # ADK
            
            "Create a professional email draft for a property inquiry. Send to: testing@example.com. Subject: Property Inquiry for 1234 Congress Ave, Austin, TX. Message: I'm interested in learning more about this property. Can you provide details about recent sales in the area and any upcoming open houses?",  # ADK
            
            "Find nearby amenities around 1234 Congress Ave, Austin, TX 78701. I need information about schools, restaurants, grocery stores, and public transportation within a 1-mile radius.",  # ADK   
            
            "Calculate travel time and distance from 1234 Congress Ave, Austin, TX to Austin-Bergstrom International Airport, and also to Dell Seton Medical Center. I need both driving and public transit options.",  # ADK
        
            "What are the key factors I should consider when buying a first home in Miami Beach, FL? Include information about market trends, property taxes, and local regulations.",  # ADK
        ]
        
        for i, query in enumerate(test_queries, 1):

            
            try:
                response = await agent.process_query(query)
            except Exception as e:
                print(f"Error: {e}")
        
        
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        if 'agent' in locals():
            await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
