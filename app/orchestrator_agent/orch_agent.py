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

PUBLIC_AGENT_CARD_PATH = "/.well-known/agent.json"


class Orchestrator:
    def __init__(self, base_urls: Dict[str, str]):
        self.base_urls = base_urls
        self.clients = {}
        self.agent_cards = {}
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    async def initialize_clients(self):
        """Initialize A2A clients for all configured agents following repo pattern."""
        self.httpx_client = httpx.AsyncClient(timeout=300.0)
        
        try:
            for name, base_url in self.base_urls.items():
                try:
                    self.logger.info(f"Initializing client for {name} at {base_url}")
                    
                    resolver = A2ACardResolver(
                        httpx_client=self.httpx_client,
                        base_url=base_url,
                    )
                    
                    final_agent_card_to_use: AgentCard | None = None
                    
                    try:
                        self.logger.info(
                            f"Fetching public agent card from: {base_url}{PUBLIC_AGENT_CARD_PATH}"
                        )
                        _public_card = await resolver.get_agent_card()
                        self.logger.info("Fetched public agent card")
                        final_agent_card_to_use = _public_card
                        
                        self.agent_cards[name] = _public_card
                        
                    except Exception as e:
                        self.logger.error(f"Error fetching public agent card: {e}")
                        raise RuntimeError(f"Failed to fetch public agent card for {name}")
                    
                    client = A2AClient(
                        httpx_client=self.httpx_client, 
                        agent_card=final_agent_card_to_use
                    )
                    
                    self.clients[name] = client
                    self.logger.info(f"A2AClient initialized for {name}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to initialize {name} client: {e}")
                    raise
        except Exception as e:
            await self.httpx_client.aclose()
            raise

    async def send_message(self, agent_name: str, text: str) -> Dict[str, Any]:
        """Send a message to an agent following the repo pattern."""
        if agent_name not in self.clients:
            available_agents = list(self.clients.keys())
            raise ValueError(f"Agent '{agent_name}' not found. Available: {available_agents}")
        
        try:
            self.logger.info(f"Sending message to {agent_name}: {text[:100]}{'...' if len(text) > 100 else ''}")
            
            client = self.clients[agent_name]
            
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
            
            self.logger.info("Sending message...")
            response = await asyncio.wait_for(client.send_message(request), timeout=300.0)
            
            self.logger.info("Response received")
            response_dict = response.model_dump()
            
            return {
                'agent': agent_name,
                'success': True,
                'response': response_dict,
                'content': self._extract_content(response_dict)
            }
            
        except asyncio.TimeoutError:
            error_message = f"Request to {agent_name} timed out after 5 minutes"
            self.logger.error(f"{error_message}")
            return {
                'agent': agent_name,
                'success': False,
                'error': 'timeout',
                'content': error_message
            }
        except Exception as e:
            error_message = f"Error communicating with {agent_name}: {str(e)}"
            self.logger.error(f"{error_message}")
            return {
                'agent': agent_name,
                'success': False,
                'error': str(e),
                'content': error_message
            }

    async def send_streaming_message(self, agent_name: str, text: str):
        """Send a streaming message to an agent."""
        if agent_name not in self.clients:
            available_agents = list(self.clients.keys())
            raise ValueError(f"Agent '{agent_name}' not found. Available: {available_agents}")
        
        try:
            self.logger.info(f"Starting streaming to {agent_name}: {text[:100]}{'...' if len(text) > 100 else ''}")
            
            client = self.clients[agent_name]
            
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
            
            self.logger.info("Starting streaming...")
            stream_response = client.send_message_streaming(streaming_request)
            
            chunk_count = 0
            start_time = asyncio.get_event_loop().time()
            timeout = 180.0  
            final_content = None
            
            async for chunk in stream_response:
                if asyncio.get_event_loop().time() - start_time > timeout:
                    self.logger.warning("Streaming timeout reached")
                    break
                    
                chunk_count += 1
                chunk_dict = chunk.model_dump()
                
                try:
                    extracted_content = self._extract_content(chunk_dict)
                    
                    if 'result' in chunk_dict and chunk_dict['result'] is not None:
                        result = chunk_dict['result']
                        if isinstance(result, dict):
                            kind = result.get('kind', 'unknown')
                            
                            if kind == 'completion':
                                if 'artifacts' in result and result['artifacts']:
                                    artifact = result['artifacts'][0]
                                    if 'parts' in artifact and artifact['parts']:
                                        part = artifact['parts'][0]
                                        if 'root' in part and 'text' in part['root']:
                                            final_content = part['root']['text']
                                            self.logger.info("Found property data in artifacts!")
                                            break
                                
                                if 'message' in result and result['message']:
                                    message = result['message']
                                    if 'parts' in message and message['parts']:
                                        part = message['parts'][0]
                                        if 'root' in part and 'text' in part['root']:
                                            final_content = part['root']['text']
                                            self.logger.info("Found property data in final message!")
                                            break
                            
                            elif kind == 'artifact-update':
                                self.logger.info("Received artifact update")
                                if 'artifact' in result and result['artifact']:
                                    artifact = result['artifact']
                                    if 'parts' in artifact and artifact['parts']:
                                        part = artifact['parts'][0]
                                        if 'text' in part and part['text']:
                                            final_content = part['text']
                                            self.logger.info("Found property data in artifact text!")
                                            break
                                        elif 'root' in part and 'text' in part['root']:
                                            final_content = part['root']['text']
                                            self.logger.info("Found property data in artifact root.text!")
                                            break
                            
                            elif kind == 'status-update' and result.get('final', False):
                                if 'message' in result and result['message']:
                                    message = result['message']
                                    if 'parts' in message and message['parts']:
                                        part = message['parts'][0]
                                        if 'root' in part and 'text' in part['root']:
                                            final_content = part['root']['text']
                                            break
                            
                            elif kind == 'status-update':
                                if 'message' in result and result['message']:
                                    message = result['message']
                                    if 'parts' in message and message['parts']:
                                        part = message['parts'][0]
                                        if 'root' in part and 'text' in part['root']:
                                            text_content = part['root']['text']
                                            if any(keyword in text_content.lower() for keyword in ['property', 'bedroom', 'bathroom', 'price', 'sqft', 'address']):
                                                final_content = text_content
                                                break
                    
                except Exception as e:
                    self.logger.error(f"Chunk #{chunk_count} extraction failed: {e}")
                    extracted_content = f"Error extracting content: {e}"
                
                yield {
                    'agent': agent_name,
                    'chunk_number': chunk_count,
                    'success': True,
                    'chunk_data': chunk_dict,
                    'content': extracted_content
                }
            
            if final_content:
                self.logger.info(f"Property data length: {len(final_content)} characters")
                yield {
                    'agent': agent_name,
                    'chunk_number': chunk_count + 1,
                    'success': True,
                    'chunk_data': {'final': True, 'content': final_content},
                    'content': final_content
                }
            
            self.logger.info(f"Streaming completed ({chunk_count} chunks)")
            
        except Exception as e:
            error_message = f"Streaming error with {agent_name}: {str(e)}"
            self.logger.error(f"{error_message}")
            yield {
                'agent': agent_name,
                'success': False,
                'error': str(e),
                'content': error_message
            }

    def _extract_content(self, response_data: Dict[str, Any]) -> str:
        """Extract readable content from response data."""
        try:
            if 'result' in response_data and response_data['result'] is not None:
                result = response_data['result']
                
                if isinstance(result, dict):
                    if 'status' in result and result['status'] is not None:
                        status = result['status']
                        if 'message' in status and status['message'] is not None:
                            message = status['message']
                            if 'parts' in message and message['parts'] is not None and len(message['parts']) > 0:
                                part = message['parts'][0]
                                if part is not None and 'root' in part and part['root'] is not None:
                                    if 'text' in part['root']:
                                        return part['root']['text']
                    
                    if 'content' in result and result['content'] is not None:
                        return str(result['content'])
                    
                    if 'message' in result and result['message'] is not None:
                        return str(result['message'])
                    
                    if 'kind' in result:
                        kind = result['kind']
                        if kind == 'status-update':
                            return "Processing your request..."
                        elif kind == 'completion':
                            return "Request completed successfully"
                        elif kind == 'error':
                            return f"Error: {result.get('error', 'Unknown error')}"
                        else:
                            return f"Status: {kind}"
            
            if 'result' in response_data and response_data['result'] is not None:
                result = response_data['result']
                if isinstance(result, dict):
                    kind = result.get('kind', 'unknown')
                    return f"Response received (type: {kind})"
            
            return "Response received (unexpected format)"
            
        except Exception as e:
            self.logger.warning(f"Content extraction error: {e}")
            return f"Response received (extraction error: {e})"

    def get_agent_info(self, agent_name: str) -> Dict[str, Any]:
        """Get information about a specific agent."""
        if agent_name not in self.agent_cards:
            return {'error': f'Agent {agent_name} not found'}
        
        card = self.agent_cards[name]
        return {
            'name': card.name,
            'description': card.description,
            'version': card.version,
            'skills': [skill.model_dump() for skill in card.skills] if card.skills else [],
            'capabilities': card.capabilities.model_dump() if card.capabilities else {},
            'url': card.url
        }

    def list_agents(self) -> Dict[str, Any]:
        """List all available agents and their capabilities."""
        agents = {}
        for name, card in self.agent_cards.items():
            agents[name] = {
                'name': card.name,
                'description': card.description,
                'skills': [skill.name for skill in card.skills] if card.skills else [],
                'url': card.url
            }
        return agents

    async def cleanup(self):
        """Clean up resources, including closing the HTTP client."""
        if hasattr(self, 'httpx_client'):
            await self.httpx_client.aclose()
            self.logger.info("HTTP client closed")

#I used this main method to test if the A2A is working and the tools are properly used
async def main():
    """Test the orchestrator with the real estate agent."""
    
    base_urls = {
        "real_estate": "http://localhost:10005", 
    }
    
    orchestrator = Orchestrator(base_urls)
    
    try:

        await orchestrator.initialize_clients()

        agents = orchestrator.list_agents()

        query = "Find 2-bedroom apartments under $3000 in Seattle"

        async for chunk in orchestrator.send_streaming_message("real_estate", query):
            if chunk['success']:
                print(f"Chunk {chunk['chunk_number']}: {chunk['content'][:200]}{'...' if len(chunk['content']) > 200 else ''}")
            else:
                print(f"Error: {chunk['content']}")
        
        print("\nTest completed!")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        await orchestrator.cleanup()


if __name__ == "__main__":
    asyncio.run(main())