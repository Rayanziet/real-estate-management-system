import os
from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, SseConnectionParams

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
    tools=[main_mcp_toolset]
)

communication_agent = LlmAgent(
    name="communication_agent",
    model="gemini-2.0-flash",
    description="Specialized agent for managing email communications in real estate. Use the gmail_create_draft_tool to create professional email drafts. Focus on client communication, property inquiries, and follow-ups. Ensure emails are professional, clear, and actionable.",
    tools=[main_mcp_toolset]
)

nearby_places_agent = LlmAgent(
    name="nearby_places_agent",
    model="gemini-2.0-flash",
    description="Specialized agent for finding nearby places and amenities around properties. Use the search_nearby_places tool to find nearby amenities like schools, hospitals, restaurants, etc. Focus on neighborhood analysis and amenity proximity for real estate decision-making. Provide comprehensive information about what's available in the surrounding area.",
    tools=[main_mcp_toolset]
)

distance_calculation_agent = LlmAgent(
    name="distance_calculation_agent",
    model="gemini-2.0-flash",
    description="Specialized agent for calculating travel times and distances between locations. Use the get_distance tool to calculate travel times and distances for different modes of transportation. Focus on commute analysis, property accessibility, and location comparison. Provide detailed travel information for real estate decision-making.",
    tools=[main_mcp_toolset]
)

root_agent = LlmAgent(
    name="real_estate_root_agent",
    model="gemini-2.0-flash",
    description="Root agent for the real estate system - main entry point and orchestrator for all queries. Your role is to: 1) Understand user queries and determine the best approach, 2) Route queries to the most appropriate specialized sub-agent: Document Analysis for document-related queries, Communication for email tasks, Nearby Places for amenities search, Distance Calculation for travel times, 3) Handle simple queries directly using the main MCP toolset, 4) Coordinate between sub-agents when complex queries require multiple capabilities, 5) Provide high-level guidance and overview of available capabilities. Available capabilities: Document analysis and RAG-based insights, Email communication and client management, Nearby amenities search and neighborhood analysis, Distance and travel time calculations. Always be helpful and guide users to the most appropriate solution.",
    tools=[main_mcp_toolset]
)

__all__ = [
    "root_agent",
    "document_analysis_agent",
    "communication_agent",
    "nearby_places_agent",
    "distance_calculation_agent",
    "main_mcp_toolset"
]

