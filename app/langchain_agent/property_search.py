import os
import warnings
from typing import Annotated, Sequence, TypedDict, Dict, Any, AsyncGenerator
import pandas as pd
import ast
from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage
)
import asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langchain_mcp_adapters.tools import load_mcp_tools
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)


load_dotenv()

REAL_ESTATE_API_KEY = os.getenv("REAL_ESTATE_API_KEY")
LLM_API_KEY = os.getenv("LLM_API_KEY")


async def load_tools():
    """Load MCP tools with timeout and retry logic for robust connection."""
    try:
        # Add timeout and retry logic for MCP connection
        sse_cm = sse_client("http://127.0.0.1:8787/sse")
        read_stream, write_stream = await asyncio.wait_for(sse_cm.__aenter__(), timeout=30.0)
        session_cm = ClientSession(read_stream, write_stream)
        session = await asyncio.wait_for(session_cm.__aenter__(), timeout=30.0)
        await asyncio.wait_for(session.initialize(), timeout=30.0)
        tools = await asyncio.wait_for(load_mcp_tools(session=session), timeout=30.0)
        tool_map = {tool.name: tool for tool in tools}

        return tools, tool_map, session_cm, sse_cm
    except asyncio.TimeoutError:
        raise Exception("MCP connection timeout - server may be overloaded")
    except Exception as e:
        raise Exception(f"Failed to connect to MCP server: {str(e)}")


class AgentState(TypedDict):
    """The state of the agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    params: dict
    properties: dict


def create_graph_sp(model: ChatGoogleGenerativeAI, system_prompt: SystemMessage, tools, tool_map):
    """Create the LangGraph workflow for property search."""
    model = model.bind_tools(tools=tools, tool_choice="auto")

    async def extract_param_node(state: AgentState):
        """Extract search parameters from user query using MCP tool."""
        tool_result = await tool_map["extract_param"].ainvoke(
            {"query": state["messages"][-1].content}
        ) 
        # Convert string response to dictionary if needed
        if isinstance(tool_result, str):
            tool_result = ast.literal_eval(tool_result)
        return {"params": tool_result}

    async def search_properties_node(state: AgentState):
        """Search for properties using extracted parameters."""
        tool_result = await tool_map["search_properties"].ainvoke({"input": state["params"]})
        # Return the raw tool result as properties, and create a simple message
        ai_message = AIMessage(content="Property search completed successfully")
        return {"properties": tool_result, "messages": [ai_message]}


    workflow = StateGraph(AgentState)
    workflow.add_node("extract_params", extract_param_node)
    workflow.add_node("search_properties", search_properties_node)
    workflow.set_entry_point("extract_params")

    workflow.add_edge("extract_params", "search_properties")
    workflow.add_edge("search_properties", END)

    return workflow.compile()


_global_tools_context = None
_tools_lock = None

async def get_or_initialize_tools():
    """Manage tool initialization to avoid re-initializing on every query."""
    global _global_tools_context, _tools_lock
    if _tools_lock is None:
        _tools_lock = asyncio.Lock()
    
    async with _tools_lock:
        if _global_tools_context is None:
            try:
                tools, tool_map, session_cm, sse_cm = await load_tools()
                _global_tools_context = (tools, tool_map, session_cm, sse_cm)
            except Exception as e:
                raise e
        return _global_tools_context


class RealEstateAgent:
    """A2A-ready Real Estate Agent"""
    
    SUPPORTED_CONTENT_TYPES = ["text/plain"]
    
    def __init__(self):
        self.conversation_states = {}  # Store states by context_id
    
    async def stream(self, query: str, context_id: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream responses in A2A format
        
        Yields:
            Dict with keys: is_task_complete, require_user_input, content
        """
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Initialize or get existing state for this conversation
                if context_id not in self.conversation_states:
                    self.conversation_states[context_id] = AgentState(
                        messages=[], params={}, properties={}
                    )
                
                state = self.conversation_states[context_id]
                
                # Yield initial working status
                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": "🔍 Analyzing your property search request..."
                }
                
                # Get tools with retry logic
                try:
                    tools, tool_map, session_cm, sse_cm = await get_or_initialize_tools()
                except Exception as e:
                    if retry_count < max_retries - 1:
                        retry_count += 1
                        yield {
                            "is_task_complete": False,
                            "require_user_input": False,
                            "content": f"⚠️ Connection issue, retrying... (attempt {retry_count}/{max_retries})"
                        }
                        await asyncio.sleep(2)  # Wait before retry
                        continue
                    else:
                        raise e
                
                # Yield parameter extraction status
                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": "📋 Extracting search parameters from your query..."
                }
                
                # Create model and system prompt
                model = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    temperature=0.3,
                    google_api_key=LLM_API_KEY
                )
                
                system_prompt = SystemMessage(
                    "You are a data extraction assistant. Your job is to extract search "
                    "parameters from user queries and retrieve raw property data. Do not "
                    "format or present the data - just return the raw results from the "
                    "search tool. Focus on accuracy and completeness of data retrieval."
                )
                
                # Add user message to state
                state["messages"].append(HumanMessage(content=query))
                
                # Create and run the graph
                graph = create_graph_sp(model, system_prompt, tools, tool_map)
                
                # Yield search status
                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": "🏠 Searching for properties matching your criteria..."
                }
                
                # Execute the agent workflow with timeout
                final_state = await asyncio.wait_for(graph.ainvoke(state), timeout=60.0)
                
                # Update stored state
                self.conversation_states[context_id] = final_state
                
                # Get the final response
                if final_state.get("properties"):
                    # Use the actual property data from the search
                    properties_data = final_state["properties"]
                    if hasattr(properties_data, 'content'):
                        # If it's an AIMessage, extract the content
                        response_content = properties_data.content
                    else:
                        # If it's raw data, format it
                        response_content = f"Property Search Results:\n\n{properties_data}"
                elif final_state["messages"] and isinstance(final_state["messages"][-1], AIMessage):
                    response_content = final_state["messages"][-1].content
                else:
                    response_content = "I couldn't process your request. Please try again."
                
                # Check if the response indicates we need more information
                needs_clarification = self._needs_user_input(response_content, final_state)
                
                if needs_clarification:
                    yield {
                        "is_task_complete": False,
                        "require_user_input": True,
                        "content": response_content + "\n\nWould you like to refine your search criteria?"
                    }
                else:
                    # Task complete - yield final response
                    yield {
                        "is_task_complete": True,
                        "require_user_input": False,
                        "content": response_content
                    }
                
                # Success - break out of retry loop
                break
                    
            except asyncio.TimeoutError:
                retry_count += 1
                if retry_count < max_retries:
                    yield {
                        "is_task_complete": False,
                        "require_user_input": False,
                        "content": f"⏰ Request timed out, retrying... (attempt {retry_count}/{max_retries})"
                    }
                    await asyncio.sleep(2)
                else:
                    yield {
                        "is_task_complete": True,
                        "require_user_input": False,
                        "content": "❌ Request timed out after multiple attempts. Please try again later."
                    }
                    break
                    
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    yield {
                        "is_task_complete": False,
                        "require_user_input": False,
                        "content": f"⚠️ Error occurred, retrying... (attempt {retry_count}/{max_retries})"
                    }
                    await asyncio.sleep(2)
                else:
                    error_msg = f"An error occurred while processing your request: {str(e)}"
                    yield {
                        "is_task_complete": True,
                        "require_user_input": False,
                        "content": error_msg
                    }
                    break
    
    def _needs_user_input(self, response: str, state: dict) -> bool:
        """
        Determine if the agent needs user input based on the response
        
        Args:
            response: The agent's response
            state: The current agent state
            
        Returns:
            True if user input is required, False otherwise
        """
        # If we have properties and they contain meaningful data, we don't need user input
        properties = state.get("properties")
        if properties is not None:
            # If properties is a string, check if it has content
            if isinstance(properties, str) and properties.strip():
                return False
            # If properties is an AIMessage, check if it has content
            elif hasattr(properties, 'content') and properties.content and properties.content.strip():
                return False
            # If properties is other data, assume it's valid
            elif properties:
                return False
        
        # Only ask for clarification if we have no properties at all
        return True

    async def cleanup(self):
        """Clean up MCP connections and resources"""
        global _global_tools_context
        if _global_tools_context:
            try:
                tools, tool_map, session_cm, sse_cm = _global_tools_context
                await session_cm.__aexit__(None, None, None)
                await sse_cm.__aexit__(None, None, None)
                _global_tools_context = None
            except Exception as e:
                pass  # Silent cleanup error


# Legacy function for backward compatibility
async def real_estate_agent(query: str, state: AgentState = None):
    """
    Legacy function - use RealEstateAgent.stream() for A2A
    """
    agent = RealEstateAgent()
    context_id = "legacy"
    
    async for item in agent.stream(query, context_id):
        if item["is_task_complete"]:
            return item["content"], agent.conversation_states.get(context_id, {})
    
    return "Processing incomplete", {}