import logging
import uuid
import httpx
import asyncio
from typing import Dict, Any

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    AgentCard,
    Message,
    MessageSendParams,
    Part,
    Role,
    SendMessageRequest,
    SendStreamingMessageRequest,
    TextPart,
)

# Constants following the repo pattern
PUBLIC_AGENT_CARD_PATH = "/.well-known/agent.json"


class ADKOrchestrator:
    """Orchestrator for managing ADK agents via A2A communication - communicates only with root_agent."""
    
    def __init__(self, root_agent_url: str):
        self.root_agent_url = root_agent_url
        self.root_client = None
        self.root_agent_card = None
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("🚀 ADK Orchestrator initialized for root_agent communication")

    async def initialize_clients(self):
        """Initialize A2A client for the root agent only."""
        # Create a persistent HTTP client that stays open with longer timeout
        self.httpx_client = httpx.AsyncClient(timeout=httpx.Timeout(300.0))  # 5 minutes timeout
        
        try:
            self.logger.info(f"🔄 Initializing client for root_agent at {self.root_agent_url}")
            
            # Initialize A2ACardResolver for root agent
            resolver = A2ACardResolver(
                httpx_client=self.httpx_client,
                base_url=self.root_agent_url,
            )
            
            try:
                self.logger.info(
                    f"📥 Fetching public agent card from: {self.root_agent_url}{PUBLIC_AGENT_CARD_PATH}"
                )
                self.root_agent_card = await resolver.get_agent_card()
                self.logger.info("✅ Fetched root agent card")
                
            except Exception as e:
                self.logger.error(f"❌ Error fetching root agent card: {e}")
                raise RuntimeError(f"Failed to fetch root agent card")
            
            # Initialize A2AClient for root agent
            self.root_client = A2AClient(
                httpx_client=self.httpx_client, 
                agent_card=self.root_agent_card
            )
            
            self.logger.info("✅ A2AClient initialized for root_agent")
            
        except Exception as e:
            # If initialization fails, close the client
            await self.httpx_client.aclose()
            raise

    async def send_message(self, text: str) -> Dict[str, Any]:
        """Send a message to the root agent, which will handle internal routing to sub-agents."""
        if not self.root_client:
            raise ValueError("❌ Root agent client not initialized. Call initialize_clients() first.")
        
        try:
            self.logger.info(f"📤 Sending message to root_agent: {text[:100]}{'...' if len(text) > 100 else ''}")
            
            # Create message payload
            message_payload = Message(
                role=Role.user,
                messageId=str(uuid.uuid4()),
                parts=[Part(root=TextPart(text=text))],
            )
            
            request = SendMessageRequest(
                id=str(uuid.uuid4()),
                params=MessageSendParams(
                    message=message_payload,
                ),
            )
            
            self.logger.info("🚀 Sending message to root_agent...")
            # Increase timeout for root agent processing (includes sub-agent coordination)
            response = await asyncio.wait_for(self.root_client.send_message(request), timeout=300.0)
            
            self.logger.info("✅ Response received from root_agent")
            response_dict = response.model_dump()
            
            # Debug: Log the response structure
            self.logger.info(f"🔍 Response structure: {response_dict}")
            
            return {
                'agent': 'root_agent',
                'agent_type': 'ADK',
                'success': True,
                'response': response_dict,
                'content': self._extract_content(response_dict)
            }
            
        except asyncio.TimeoutError:
            error_message = f"Request to root_agent timed out after 3 minutes"
            self.logger.error(f"❌ {error_message}")
            return {
                'agent': 'root_agent',
                'agent_type': 'ADK',
                'success': False,
                'error': 'timeout',
                'content': error_message
            }
        except Exception as e:
            error_message = f"Error communicating with root_agent: {str(e)}"
            self.logger.error(f"❌ {error_message}")
            
            # Add more detailed error logging
            if hasattr(e, 'response') and e.response:
                self.logger.error(f"🔍 HTTP Response Status: {e.response.status_code}")
                self.logger.error(f"🔍 HTTP Response Headers: {e.response.headers}")
                try:
                    response_text = e.response.text
                    self.logger.error(f"🔍 HTTP Response Body: {response_text[:500]}...")
                except:
                    self.logger.error("🔍 Could not read response body")
            
            return {
                'agent': 'root_agent',
                'agent_type': 'ADK',
                'success': False,
                'error': str(e),
                'content': error_message
            }

    async def send_streaming_message(self, text: str):
        """Send a streaming message to the root agent."""
        if not self.root_client:
            raise ValueError("❌ Root agent client not initialized. Call initialize_clients() first.")
        
        try:
            self.logger.info(f"🔄 Starting streaming to root_agent: {text[:100]}{'...' if len(text) > 100 else ''}")
            
            # Create message payload
            message_payload = Message(
                role=Role.user,
                messageId=str(uuid.uuid4()),
                parts=[Part(root=TextPart(text=text))],
            )
            
            streaming_request = SendStreamingMessageRequest(
                id=str(uuid.uuid4()),
                params=MessageSendParams(
                    message=message_payload,
                ),
            )
            
            self.logger.info("🌊 Starting streaming...")
            stream_response = self.root_client.send_message_streaming(streaming_request)
            
            chunk_count = 0
            start_time = asyncio.get_event_loop().time()
            timeout = 240.0  # 4 minutes timeout for streaming (includes sub-agent coordination)
            final_content = None
            
            async for chunk in stream_response:
                # Check timeout
                if asyncio.get_event_loop().time() - start_time > timeout:
                    self.logger.warning("⚠️ Streaming timeout reached")
                    break
                    
                chunk_count += 1
                chunk_dict = chunk.model_dump()
                
                # Extract content and check for completion messages
                try:
                    extracted_content = self._extract_content(chunk_dict)
                    
                    # Check if this is a completion message
                    if 'result' in chunk_dict and chunk_dict['result'] is not None:
                        result = chunk_dict['result']
                        if isinstance(result, dict):
                            kind = result.get('kind', 'unknown')
                            
                            # Check for completion status
                            if kind == 'completion':
                                self.logger.info("✅ Received completion message from root_agent")
                                # Look for artifacts or final content
                                if 'artifacts' in result and result['artifacts']:
                                    artifact = result['artifacts'][0]
                                    if 'parts' in artifact and artifact['parts']:
                                        part = artifact['parts'][0]
                                        if 'root' in part and 'text' in part['root']:
                                            final_content = part['root']['text']
                                            self.logger.info("📋 Found consolidated response data in artifacts!")
                                            break
                                
                                # Check for final message
                                if 'message' in result and result['message']:
                                    message = result['message']
                                    if 'parts' in message and message['parts']:
                                        part = message['parts'][0]
                                        if 'root' in part and 'text' in part['root']:
                                            final_content = part['root']['text']
                                            self.logger.info("📋 Found consolidated response data in final message!")
                                            break
                            
                            # Check for artifact updates
                            elif kind == 'artifact-update':
                                self.logger.info("📦 Received artifact update from root_agent")
                                if 'artifact' in result and result['artifact']:
                                    artifact = result['artifact']
                                    if 'parts' in artifact and artifact['parts']:
                                        part = artifact['parts'][0]
                                        # Check for text field
                                        if 'text' in part and part['text']:
                                            final_content = part['text']
                                            self.logger.info("📋 Found consolidated response data in artifact text!")
                                            break
                                        # Fallback to root.text structure
                                        elif 'root' in part and 'text' in part['root']:
                                            final_content = part['root']['text']
                                            self.logger.info("📋 Found consolidated response data in artifact root.text!")
                                            break
                            
                            # Check for final status updates
                            elif kind == 'status-update' and result.get('final', False):
                                self.logger.info("✅ Received final status update from root_agent")
                                if 'message' in result and result['message']:
                                    message = result['message']
                                    if 'parts' in message and message['parts']:
                                        part = message['parts'][0]
                                        if 'root' in part and 'text' in part['root']:
                                            final_content = part['root']['text']
                                            self.logger.info("📋 Found consolidated response data in final status!")
                                            break
                    
                except Exception as e:
                    self.logger.error(f"❌ Chunk #{chunk_count} extraction failed: {e}")
                    extracted_content = f"Error extracting content: {e}"
                
                yield {
                    'agent': 'root_agent',
                    'agent_type': 'ADK',
                    'chunk_number': chunk_count,
                    'success': True,
                    'chunk_data': chunk_dict,
                    'content': extracted_content
                }
            
            # If we found final content, yield it as the last chunk
            if final_content:
                self.logger.info("🎯 Yielding final consolidated response data")
                self.logger.info(f"📊 Response data length: {len(final_content)} characters")
                yield {
                    'agent': 'root_agent',
                    'agent_type': 'ADK',
                    'chunk_number': chunk_count + 1,
                    'success': True,
                    'chunk_data': {'final': True, 'content': final_content},
                    'content': final_content
                }
            
            self.logger.info(f"✅ Streaming completed ({chunk_count} chunks)")
            
        except Exception as e:
            error_message = f"Streaming error with root_agent: {str(e)}"
            self.logger.error(f"❌ {error_message}")
            yield {
                'agent': 'root_agent',
                'agent_type': 'ADK',
                'success': False,
                'error': str(e),
                'content': error_message
            }

    def _extract_content(self, response_data: Dict[str, Any]) -> str:
        """Extract readable content from response data."""
        try:
            # For A2A responses, look for the actual content in the result structure
            if 'result' in response_data and response_data['result'] is not None:
                result = response_data['result']
                
                # Check if this is a status update with actual content
                if isinstance(result, dict):
                    if 'status' in result and result['status'] is not None:
                        status = result['status']
                        if 'message' in status and status['message'] is not None:
                            # Extract the actual message content
                            message = status['message']
                            if 'parts' in message and message['parts'] is not None and len(message['parts']) > 0:
                                # Get the text content from the message parts
                                part = message['parts'][0]
                                if part is not None and 'root' in part and part['root'] is not None:
                                    if 'text' in part['root']:
                                        return part['root']['text']
                    
                    # Check for direct content in result
                    if 'content' in result and result['content'] is not None:
                        return str(result['content'])
                    
                    # Check for message in result
                    if 'message' in result and result['message'] is not None:
                        return str(result['message'])
                    
                    # Check for kind and provide meaningful status
                    if 'kind' in result:
                        kind = result['kind']
                        if kind == 'status-update':
                            return "🔄 Root agent is processing your request and coordinating with sub-agents..."
                        elif kind == 'completion':
                            return "✅ Request completed successfully by root agent and sub-agents"
                        elif kind == 'error':
                            return f"❌ Error: {result.get('error', 'Unknown error')}"
                        else:
                            return f"📋 Status: {kind}"
            
            # If we can't extract meaningful content, return a summary
            if 'result' in response_data and response_data['result'] is not None:
                result = response_data['result']
                if isinstance(result, dict):
                    kind = result.get('kind', 'unknown')
                    return f"Response received from root agent (type: {kind})"
            
            # Fallback for unexpected structures
            return "Response received from root agent (unexpected format)"
            
        except Exception as e:
            self.logger.warning(f"⚠️ Content extraction error: {e}")
            # Return a safe fallback instead of crashing
            return f"Response received from root agent (extraction error: {e})"

    def get_root_agent_info(self) -> Dict[str, Any]:
        """Get information about the root agent."""
        if not self.root_agent_card:
            return {'error': 'Root agent card not available'}
        
        return {
            'name': self.root_agent_card.name,
            'description': self.root_agent_card.description,
            'version': self.root_agent_card.version,
            'type': 'ADK Root Agent',
            'skills': [skill.model_dump() for skill in self.root_agent_card.skills] if self.root_agent_card.skills else [],
            'capabilities': self.root_agent_card.capabilities.model_dump() if self.root_agent_card.capabilities else {},
            'url': self.root_agent_card.url
        }

    async def cleanup(self):
        """Clean up resources, including closing the HTTP client."""
        if hasattr(self, 'httpx_client'):
            await self.httpx_client.aclose()
            self.logger.info("🧹 HTTP client closed")


# ==== Test Function ====
async def main():
    """Test the ADK Orchestrator with root_agent communication."""
    
    # Configure root agent endpoint (this needs to be running as an A2A server)
    root_agent_url = "http://localhost:10002"  # ADK root agent port (matches __main__.py)
    
    orchestrator = ADKOrchestrator(root_agent_url)
    
    try:
        print("🚀 Starting ADK Root Agent Orchestrator Test")
        print("=" * 50)
        print("Note: Make sure the root_agent is running as an A2A server on port 10002")
        print("The root_agent will handle internal routing to sub-agents")
        print("=" * 50)
        
        # Initialize client
        await orchestrator.initialize_clients()
        print("\n✅ Root agent client initialized!")
        
        # Show root agent info
        print("\n📋 Root Agent Information:")
        agent_info = orchestrator.get_root_agent_info()
        print(f"  • Name: {agent_info['name']}")
        print(f"  • Type: {agent_info['type']}")
        print(f"  • Description: {agent_info['description'][:100]}...")
        if 'skills' in agent_info and agent_info['skills']:
            print(f"  • Skills: {len(agent_info['skills'])} available")
        
        # Test simple message to root agent
        print("\n" + "="*50)
        print("🧪 TEST: Testing each ADK tool capability")
        print("="*50)
        
        # Test 1: Document Analysis
        print("\n📄 TEST 1: Document Analysis")
        print("-" * 30)
        doc_query = "Analyze this property tax document: The property at 123 Main St, Anytown, CA has an assessed value of $450,000 with annual property taxes of $5,400. The tax rate is 1.2% and includes school district, county, and city levies."
        print(f"Query: {doc_query}")
        
        try:
            response = await orchestrator.send_message(doc_query)
            if response['success']:
                print(f"✅ Document Analysis Response: {response['content'][:300]}...")
            else:
                print(f"❌ Error: {response['content']}")
        except Exception as e:
            print(f"❌ Exception: {e}")
        
        # Test 2: Email Communication
        print("\n📧 TEST 2: Email Communication")
        print("-" * 30)
        email_query = "Create a professional email draft for a property inquiry. Send to: testing@example.com. Subject: Property Inquiry for 123 Main St. Message: I'm interested in learning more about this property. Can you provide details about recent sales in the area and any upcoming open houses?"
        print(f"Query: {email_query}")
        
        try:
            response = await orchestrator.send_message(email_query)
            if response['success']:
                print(f"✅ Email Communication Response: {response['content'][:300]}...")
            else:
                print(f"❌ Error: {response['content']}")
        except Exception as e:
            print(f"❌ Exception: {e}")
        
        # Test 3: Nearby Places Search
        print("\n🏠 TEST 3: Nearby Places Search")
        print("-" * 30)
        nearby_query = "Find nearby amenities around 123 Main St, Anytown, CA. I need information about schools, restaurants, grocery stores, and public transportation within a 1-mile radius."
        print(f"Query: {nearby_query}")
        
        try:
            response = await orchestrator.send_message(nearby_query)
            if response['success']:
                print(f"✅ Nearby Places Response: {response['content'][:300]}...")
            else:
                print(f"❌ Error: {response['content']}")
        except Exception as e:
            print(f"❌ Exception: {e}")
        
        # Test 4: Distance Calculation
        print("\n🚗 TEST 4: Distance Calculation")
        print("-" * 30)
        distance_query = "Calculate travel time and distance from 123 Main St, Anytown, CA to downtown Anytown, and also to the nearest hospital. I need both driving and public transit options."
        print(f"Query: {distance_query}")
        
        try:
            response = await orchestrator.send_message(distance_query)
            if response['success']:
                print(f"✅ Distance Calculation Response: {response['content'][:300]}...")
            else:
                print(f"❌ Error: {response['content']}")
        except Exception as e:
            print(f"❌ Exception: {e}")
        
        # Test 5: General Real Estate Guidance
        print("\n🏡 TEST 5: General Real Estate Guidance")
        print("-" * 30)
        general_query = "What are the key factors I should consider when buying a first home in Anytown, CA? Include information about market trends, property taxes, and local regulations."
        print(f"Query: {general_query}")
        
        try:
            response = await orchestrator.send_message(general_query)
            if response['success']:
                print(f"✅ General Guidance Response: {response['content'][:300]}...")
            else:
                print(f"❌ Error: {response['content']}")
        except Exception as e:
            print(f"❌ Exception: {e}")
        
        print("\n✅ Test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Always cleanup resources
        await orchestrator.cleanup()
        print("🧹 Cleanup completed")


if __name__ == "__main__":
    asyncio.run(main())
