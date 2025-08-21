from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import asyncio
import logging
from simple_agent import SimpleAgent
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Real Estate Agent API",
    description="API for Real Estate Agent with LLM Integration",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your Streamlit app domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = "default_user"

class ChatResponse(BaseModel):
    response: str
    orchestrator_used: str
    success: bool
    error: Optional[str] = None

class StatusResponse(BaseModel):
    agent_status: str
    langchain_status: str
    adk_status: str
    llm_status: str

# Global agent instance
agent: Optional[SimpleAgent] = None

@app.on_event("startup")
async def startup_event():
    """Initialize the agent on startup."""
    global agent
    try:
        # Get configuration from environment variables
        langchain_url = os.getenv("LANGCHAIN_URL", "http://localhost:10005")
        adk_url = os.getenv("ADK_URL", "http://localhost:10002")
        
        logger.info(f"Initializing agent with LangChain: {langchain_url}, ADK: {adk_url}")
        
        # Initialize the simple agent
        agent = SimpleAgent(langchain_url, adk_url)
        await agent.initialize()
        
        logger.info("✅ Agent initialized successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize agent: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global agent
    if agent:
        await agent.cleanup()
        logger.info("🧹 Agent cleanup completed")

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Real Estate Agent API is running!"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": asyncio.get_event_loop().time()}

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get agent status."""
    global agent
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        status = await agent.get_status()
        # Add LLM status
        status['llm_status'] = 'available'  # We'll enhance this later
        return StatusResponse(**status)
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting status: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat message through the agent."""
    global agent
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        logger.info(f"Processing chat request: {request.message}")
        
        # Process the query through the simple agent
        response = await agent.process_query(request.message)
        
        # Format the response for the frontend
        if response.get('success', False):
            return ChatResponse(
                response=response.get('formatted_content', 'No content available'),
                orchestrator_used=response.get('orchestrator_used', 'unknown'),
                success=True
            )
        else:
            return ChatResponse(
                response=response.get('formatted_content', 'Error processing request'),
                orchestrator_used=response.get('orchestrator_used', 'unknown'),
                success=False,
                error=response.get('error', 'Unknown error')
            )
            
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream chat responses for real-time updates."""
    global agent
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        logger.info(f"Processing streaming chat request: {request.message}")
        
        # For now, we'll return a single response
        # Later we can implement actual streaming
        response = await agent.process_query(request.message)
        
        if response.get('success', False):
            return {
                "response": response.get('formatted_content', 'No content available'),
                "orchestrator_used": response.get('orchestrator_used', 'unknown'),
                "success": True
            }
        else:
            return {
                "response": response.get('formatted_content', 'Error processing request'),
                "orchestrator_used": response.get('orchestrator_used', 'unknown'),
                "success": False,
                "error": response.get('error', 'Unknown error')
            }
            
    except Exception as e:
        logger.error(f"Error processing streaming chat request: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
