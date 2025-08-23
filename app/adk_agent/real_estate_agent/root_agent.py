import os
from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, SseConnectionParams
from google.adk.runners import Runner
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.sessions import InMemorySessionService

main_mcp_toolset = MCPToolset(
    connection_params=SseConnectionParams(
        url="http://127.0.0.1:8001/sse",
        timeout=30.0 
    )
)

document_analysis_agent = LlmAgent(
    name="document_analysis_agent",
    model="gemini-2.0-flash",
    description="Specialized agent for analyzing real estate documents using RAG pipeline. Use the rag_pipeline_tool to retrieve relevant information from documents. Focus on property tax information, market analysis, and regulatory compliance. Provide clear, actionable insights based on the retrieved information.",
    instruction="You are a specialized real estate document analysis agent. Your role is to analyze documents, extract key information, and provide insights using the RAG pipeline tools. Always be thorough and provide actionable recommendations.",
    tools=[main_mcp_toolset]
)

communication_agent = LlmAgent(
    name="communication_agent",
    model="gemini-2.0-flash",
    description="Specialized agent for managing email communications in real estate. Use the gmail_create_draft_tool to create professional email drafts. Focus on client communication, property inquiries, and follow-ups. Ensure emails are professional, clear, and actionable.",
    instruction="You are a specialized real estate communication agent. Your role is to create professional email drafts and manage client communications. Always maintain a professional tone and ensure clarity in all communications.",
    tools=[main_mcp_toolset]
)

nearby_places_agent = LlmAgent(
    name="nearby_places_agent",
    model="gemini-2.0-flash",
    description="Specialized agent for finding nearby places and amenities around properties. Use the search_nearby_places tool to find nearby amenities like schools, hospitals, restaurants, etc. Focus on neighborhood analysis and amenity proximity for real estate decision-making. Provide comprehensive information about what's available in the surrounding area.",
    instruction="You are a specialized real estate location analysis agent. Your role is to find and analyze nearby amenities and places around properties. Provide comprehensive neighborhood insights to help with real estate decisions.",
    tools=[main_mcp_toolset]
)

distance_calculation_agent = LlmAgent(
    name="distance_calculation_agent",
    model="gemini-2.0-flash",
    description="Specialized agent for calculating travel times and distances between locations. Use the get_distance tool to calculate travel times and distances for different modes of transportation. Focus on commute analysis, property accessibility, and location comparison. Provide detailed travel information for real estate decision-making.",
    instruction="You are a specialized real estate travel analysis agent. Your role is to calculate travel times and distances between locations. Provide detailed commute and accessibility information for real estate decision-making.",
    tools=[main_mcp_toolset]
)

root_agent = LlmAgent(
    name="real_estate_root_agent",
    model="gemini-2.0-flash",
    description="Root agent for the real estate system - main entry point and orchestrator for all queries. Your role is to: 1) Understand user queries and determine the best approach, 2) Route queries to the most appropriate specialized sub-agent: Document Analysis for document-related queries, Communication for email tasks, Nearby Places for amenities search, Distance Calculation for travel times, 3) Coordinate between sub-agents when complex queries require multiple capabilities, 4) Provide high-level guidance and overview of available capabilities. Available capabilities: Document analysis and RAG-based insights, Email communication and client management, Nearby amenities search and neighborhood analysis, Distance and travel time calculations. Always be helpful and guide users to the most appropriate solution.",
    instruction="You are the main orchestrator for the real estate system. Your role is to understand user queries, route them to appropriate specialized agents, and coordinate complex multi-agent workflows. Always provide clear guidance and ensure users get the best possible assistance. When a user asks about document analysis, route to document_analysis_agent. For email communications, route to communication_agent. For nearby places and amenities, route to nearby_places_agent. For distance and travel calculations, route to distance_calculation_agent. Coordinate between agents when queries require multiple capabilities.",
    sub_agents=[document_analysis_agent, communication_agent, nearby_places_agent, distance_calculation_agent]
)

def create_runner(agent):
    """Create a Runner instance for the given agent."""
    return Runner(
        app_name=agent.name,
        agent=agent,
        artifact_service=InMemoryArtifactService(),#used for storing what agents generate and sharing them temporarily between agents.
        session_service=InMemorySessionService(),#manages conversation sessions and context
        memory_service=InMemoryMemoryService(),#stores conversation history and agent memory
    )

root_agent_runner = create_runner(root_agent)

__all__ = [
    "root_agent",
    "document_analysis_agent",
    "communication_agent",
    "nearby_places_agent",
    "distance_calculation_agent",
    "main_mcp_toolset",
    "root_agent_runner"
]

