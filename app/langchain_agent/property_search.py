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
import json
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


def create_graph_sp(model : ChatGoogleGenerativeAI, system_prompt: SystemMessage, tools, tool_map):
    
    model = model.bind_tools(tools=tools, tool_choice="auto")

    async def extract_param_node(state: AgentState):
        tool_result = await tool_map["extract_param"].ainvoke(
            {"query": state["messages"][-1].content}
        )
        #this is crucial to make sure the response is dictionary
        if isinstance(tool_result, str):
            tool_result = ast.literal_eval(tool_result)
        state["params"] = tool_result
        return {"params": tool_result}

    async def search_properties_node(state: AgentState):
        if "params" not in state or not isinstance(state["params"], dict):
            raise ValueError(f"state['params'] is missing or not a dict: {state.get('params')}")
        tool_result = await tool_map["search_properties"].ainvoke({"input": state["params"]})
        # this way will help the model maintains full conversation memory, even the ai responses
        ai_message = AIMessage(content=tool_result)
        state["properties"] = tool_result
        return {"properties": tool_result, "messages": [ai_message]}

    # def should_continue(state: AgentState):
    #     messages = state["messages"]
    #     last_message = messages[-1]
    #     if not last_message.tool_calls:
    #         return "end"
    #     else:
    #         return "continue"
        
    workflow = StateGraph(AgentState)
    workflow.add_node("extract_params", extract_param_node)
    workflow.add_node("search_properties", search_properties_node)

    workflow.set_entry_point("extract_params")
    workflow.add_edge("extract_params", "search_properties")
    workflow.add_edge("search_properties", END)

    #workflow.add_conditional_edge("search_properties", "end", should_continue)
    graph = workflow.compile()
    return graph


async def real_estate_agent(query: str, state: AgentState = None):
    try:
        tools, tool_map, session_cm, sse_cm = await load_tools()
        model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
            google_api_key=LLM_API_KEY
        )
        system_prompt = SystemMessage("You are a professional real estate assistant for RentCast properties. Understand the user's property search queries, extract relevant details, and generate clear, friendly, and accurate responses. Focus on the requested location, price, property type, and features. Present results in a readable, engaging way like a human real estate agent. Highlight key information: address, bedrooms, bathrooms, square footage, and listing price. Summarize up to 5 top matches. If no results, suggest adjusting the search criteria. Do not fabricate information or assume missing values. Return plain text only; use a warm, helpful, and professional tone.")


        if state is None: #here we are initializing the state
            state = AgentState(messages=[], params={}, properties={})

        state["messages"].append(HumanMessage(content=query))
        #appending the user query to the state, this will help the tools maintain a full conversation history

        graph = create_graph_sp(model, system_prompt, tools, tool_map)

        final_state = await graph.ainvoke(state)
            
            
        if final_state["messages"] and isinstance(final_state["messages"][-1], AIMessage):
            return final_state["messages"][-1].content, final_state
        else:
            return "I couldn't process your request. Please try again.", final_state
    except Exception as e:
        print(f"Error occurred: {e}")
        return "An error occurred while processing your request.", state
    


if __name__ == "__main__":
    import asyncio

    async def test_agent():
        query = "I'm looking for a 3-bedroom, 2-bathroom house in Miami under $1,500,000."
        response, state = await real_estate_agent(query)
        print("\n=== AI Response ===\n")
        print(response)
        print("\n=== Conversation State ===\n")
        print(state)

    asyncio.run(test_agent())

