# adk_agent.py
import os
from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, SseConnectionParams



root_agent = LlmAgent(
    name="real_estate_agent",
    model="gemini-2.5-flash",
    tools=[
        MCPToolset(
            connection_params=SseConnectionParams(
                url="http://127.0.0.1:8001/sse",
                timeout=30.0 
            )
        )
    ],
)

# Optional: Add error handling for agent initialization
if __name__ == "__main__":
    print("Real Estate ADK Agent initialized successfully!")
    print("MCP Tools configured for: http://127.0.0.1:8000/sse")
    print("Make sure your MCP server is running before starting the ADK web interface.")