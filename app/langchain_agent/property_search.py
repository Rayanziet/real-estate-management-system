import os
import warnings
from typing import Annotated, Sequence, TypedDict
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
    sse_cm = sse_client("http://127.0.0.1:8787/sse")
    read_stream, write_stream = await sse_cm.__aenter__()
    session_cm = ClientSession(read_stream, write_stream)
    session = await session_cm.__aenter__()
    await session.initialize()
    tools = await load_mcp_tools(session=session)
    tool_map = {tool.name: tool for tool in tools}

    return tools, tool_map, session_cm, sse_cm


class AgentState(TypedDict):
    """The state of the agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    params: dict
    properties: dict


def create_graph_sp(model: ChatGoogleGenerativeAI, system_prompt: SystemMessage, tools, tool_map):
    model = model.bind_tools(tools=tools, tool_choice="auto")

    async def extract_param_node(state: AgentState):
        tool_result = await tool_map["extract_param"].ainvoke(
            {"query": state["messages"][-1].content}
        ) 
        #this is crucial to make sure the response is dictionary
        if isinstance(tool_result, str):
            tool_result = ast.literal_eval(tool_result)
        # state["params"] = tool_result
        return {"params": tool_result}

    async def search_properties_node(state: AgentState):
        tool_result = await tool_map["search_properties"].ainvoke({"input": state["params"]})
        # this way will help the model maintains full conversation memory, even the ai responses
        ai_message = AIMessage(content=tool_result)
        # state["properties"] = tool_result
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
#this function is essential for managing tool initialization
#without it, the tools will be initialized on every query
async def get_or_initialize_tools():
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


async def real_estate_agent(query: str, state: AgentState = None):
    """
    A2A-ready real estate agent function
    Returns: (response_text, full_state_dict) for orchestrator use
    """
    try:
        tools, tool_map, session_cm, sse_cm = await get_or_initialize_tools()
            
        model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
            google_api_key=LLM_API_KEY
        )
        system_prompt = SystemMessage("You are a data extraction assistant. Your job is to extract search parameters from user queries and retrieve raw property data. Do not format or present the data - just return the raw results from the search tool. Focus on accuracy and completeness of data retrieval.")
        if state is None: #here we are initializing the state
            state = AgentState(messages=[], params={}, properties={})

        state["messages"].append(HumanMessage(content=query))
        #appending the user query to the state, this will help the tools maintain a full conversation history

        graph = create_graph_sp(model, system_prompt, tools, tool_map)

        final_state = await graph.ainvoke(state)
            
        if final_state["messages"] and isinstance(final_state["messages"][-1], AIMessage):
            return final_state["messages"][-1].content, dict(final_state)
        else:
            return "I couldn't process your request. Please try again.", dict(final_state)

    except Exception as e:
        print(f"Detailed error in real_estate_agent: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        
        if state is None:
            state = AgentState(messages=[], params={}, properties={})
        return f"An error occurred while processing your request: {str(e)}", dict(state)
