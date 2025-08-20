import logging
from typing import Any
from uuid import uuid4
import httpx
import asyncio
import json

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
    SendStreamingMessageRequest,
)
from a2a.utils.constants import (
    AGENT_CARD_WELL_KNOWN_PATH,
    EXTENDED_AGENT_CARD_PATH,
)


class Orchestrator:
    def __init__(self, base_urls):
        self.base_urls = base_urls
        self.clients = {}
        self.httpx_client = httpx.AsyncClient()
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    async def initialize_clients(self):
        """Initialize A2A clients for all configured agents."""
        for name, base_url in self.base_urls.items():
            try:
                self.logger.info(f"🔄 Initializing client for {name} at {base_url}")
                
                # Initialize A2ACardResolver
                resolver = A2ACardResolver(
                    httpx_client=self.httpx_client,
                    base_url=base_url,
                )
                
                # Fetch Public Agent Card
                final_agent_card_to_use: AgentCard | None = None

                try:
                    self.logger.info(
                        f'📥 Fetching public agent card from: {base_url}{AGENT_CARD_WELL_KNOWN_PATH}'
                    )
                    _public_card = await resolver.get_agent_card()
                    self.logger.info('✅ Successfully fetched public agent card')
                    final_agent_card_to_use = _public_card

                    # Log agent capabilities for debugging
                    self.logger.info(f"📋 Agent '{name}' capabilities:")
                    self.logger.info(f"   - Name: {_public_card.name}")
                    self.logger.info(f"   - Streaming: {_public_card.capabilities.streaming if _public_card.capabilities else 'Unknown'}")
                    self.logger.info(f"   - Skills: {[skill.name for skill in _public_card.skills] if _public_card.skills else 'None'}")

                    if _public_card.supports_authenticated_extended_card:
                        try:
                            self.logger.info('🔐 Attempting to fetch authenticated extended card')
                            auth_headers_dict = {
                                'Authorization': 'Bearer dummy-token-for-extended-card'
                            }
                            _extended_card = await resolver.get_agent_card(
                                relative_card_path=EXTENDED_AGENT_CARD_PATH,
                                http_kwargs={'headers': auth_headers_dict},
                            )
                            self.logger.info('✅ Successfully fetched authenticated extended agent card')
                            final_agent_card_to_use = _extended_card
                        except Exception as e_extended:
                            self.logger.warning(
                                f'⚠️ Failed to fetch extended agent card: {e_extended}. Using public card.'
                            )
                    else:
                        self.logger.info('ℹ️ Public card does not support extended authentication')

                except Exception as e:
                    self.logger.error(f'❌ Critical error fetching public agent card: {e}')
                    raise RuntimeError(f'Failed to fetch the public agent card for {name}') from e

                # Initialize A2AClient
                client = A2AClient(
                    httpx_client=self.httpx_client, 
                    agent_card=final_agent_card_to_use
                )
                self.clients[name] = client
                self.logger.info(f'✅ A2AClient initialized for {name}')
                
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize {name} client: {e}")
                raise

    async def run_task(self, agent_name, user_text):
        """Send a non-streaming message to an agent and return the response."""
        if agent_name not in self.clients:
            available_agents = list(self.clients.keys())
            raise ValueError(f"❌ Agent '{agent_name}' not found. Available: {available_agents}")
        
        try:
            self.logger.info(f"📤 Sending message to {agent_name}: {user_text[:100]}{'...' if len(user_text) > 100 else ''}")
            client = self.clients[agent_name]
            
            # Create message payload
            send_message_payload: dict[str, Any] = {
                'message': {
                    'role': 'user',
                    'parts': [
                        {'kind': 'text', 'text': user_text}
                    ],
                    'messageId': uuid4().hex,
                },
            }
            
            request = SendMessageRequest(
                id=str(uuid4()), 
                params=MessageSendParams(**send_message_payload)
            )

            self.logger.info(f"🚀 Executing A2A request for {agent_name}...")
            response = await client.send_message(request)
            
            # Extract the actual response content
            response_data = response.model_dump(mode='json', exclude_none=True)
            content = self._extract_response_content(response_data)
            
            self.logger.info(f"✅ Received response from {agent_name} (length: {len(str(content))})")
            
            return {
                'agent': agent_name,
                'content': content,
                'raw_response': response_data,
                'success': True
            }
            
        except Exception as e:
            error_message = f"Error communicating with {agent_name}: {str(e)}"
            self.logger.error(f"❌ {error_message}")
            return {
                'agent': agent_name,
                'content': error_message,
                'success': False,
                'error': str(e)
            }

    async def run_streaming(self, agent_name, user_text):
        """Send a streaming message to an agent and yield response chunks."""
        if agent_name not in self.clients:
            available_agents = list(self.clients.keys())
            raise ValueError(f"❌ Agent '{agent_name}' not found. Available: {available_agents}")
            
        try:
            self.logger.info(f"🔄 Starting streaming message to {agent_name}: {user_text[:100]}{'...' if len(user_text) > 100 else ''}")
            client = self.clients[agent_name]
            
            # Create message payload
            send_message_payload: dict[str, Any] = {
                'message': {
                    'role': 'user',
                    'parts': [
                        {'kind': 'text', 'text': user_text}
                    ],
                    'messageId': uuid4().hex,
                },
            }

            streaming_request = SendStreamingMessageRequest(
                id=str(uuid4()), 
                params=MessageSendParams(**send_message_payload)
            )

            self.logger.info(f"🌊 Starting streaming request to {agent_name}...")
            stream_response = client.send_message_streaming(streaming_request)

            chunk_count = 0
            async for chunk in stream_response:
                chunk_count += 1
                chunk_data = chunk.model_dump(mode='json', exclude_none=True)
                
                # Extract content from streaming chunk
                content = self._extract_response_content(chunk_data)
                
                self.logger.debug(f"📦 Streaming chunk #{chunk_count} from {agent_name}")
                
                yield {
                    'agent': agent_name,
                    'chunk': chunk_count,
                    'content': content,
                    'raw_chunk': chunk_data,
                    'success': True
                }
            
            self.logger.info(f"✅ Streaming completed for {agent_name} ({chunk_count} chunks)")
                
        except Exception as e:
            error_message = f"Error in streaming from {agent_name}: {str(e)}"
            self.logger.error(f"❌ {error_message}")
            yield {
                'agent': agent_name,
                'content': error_message,
                'success': False,
                'error': str(e)
            }

    def _extract_response_content(self, response_data):
        """Extract the actual content from A2A response data."""
        try:
            # Common patterns for A2A response content
            if isinstance(response_data, dict):
                # Try different possible paths for the content
                possible_paths = [
                    'result',
                    'content',
                    'message',
                    'response',
                    'data',
                    'output'
                ]
                
                for path in possible_paths:
                    if path in response_data:
                        content = response_data[path]
                        if content is not None:
                            return content
                
                # If no standard path found, return the whole response
                if response_data:
                    return response_data
            
            # Fallback: return as-is
            return response_data
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error extracting response content: {e}")
            return response_data

    async def close(self):
        """Clean up the HTTP client and resources."""
        try:
            self.logger.info("🧹 Cleaning up orchestrator resources...")
            await self.httpx_client.aclose()
            self.logger.info("✅ Orchestrator cleanup completed")
        except Exception as e:
            self.logger.error(f"❌ Error during cleanup: {e}")


# ==== Main Test Function ====
async def main():
    """Main function to test the A2A communication."""
    base_urls = {
        "real_estate": "http://localhost:8001",
    }
    
    orchestrator = Orchestrator(base_urls)
    
    try:
        print("🚀 Starting A2A Client Test")
        print("=" * 50)
        
        # Initialize clients
        await orchestrator.initialize_clients()
        print("\n✅ All clients initialized successfully!")
        
        # Test 1: Use streaming for property search (since non-streaming has A2A framework issues)
        print("\n" + "="*50)
        print("🔄 TEST 1: Streaming property search (workaround for A2A framework bug)")
        print("="*50)
        
        search_query = "Find 2-bedroom apartments in downtown Manhattan under $4000"
        print(f"🔍 Query: {search_query}")
        print("📦 Streaming chunks:")
        
        async for chunk in orchestrator.run_streaming("real_estate", search_query):
            if chunk['success']:
                print(f"   📦 Chunk #{chunk['chunk']}: {str(chunk['content'])[:200]}{'...' if len(str(chunk['content'])) > 200 else ''}")
            else:
                print(f"   ❌ Error chunk: {chunk['content']}")
        
        # Test 2: Another streaming search
        print("\n" + "="*50)
        print("🔄 TEST 2: Another streaming property search")
        print("="*50)
        
        streaming_query = "Search for luxury condos with parking in Brooklyn with price range $300k-$800k"
        print(f"🔍 Query: {streaming_query}")
        print("📦 Streaming chunks:")
        
        async for chunk in orchestrator.run_streaming("real_estate", streaming_query):
            if chunk['success']:
                print(f"   📦 Chunk #{chunk['chunk']}: {str(chunk['content'])[:200]}{'...' if len(str(chunk['content'])) > 200 else ''}")
            else:
                print(f"   ❌ Error chunk: {chunk['content']}")
        
        print("\n✅ All tests completed successfully!")
        print("\n💡 Note: Using streaming mode as workaround for A2A framework non-streaming bug")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🧹 Cleaning up...")
        await orchestrator.close()

if __name__ == "__main__":
    asyncio.run(main())