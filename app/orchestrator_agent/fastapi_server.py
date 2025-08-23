from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import asyncio
import logging
from simple_agent import SimpleAgent
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Real Estate Agent API",
    description="API for Real Estate Agent with LLM Integration",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

agent: Optional[SimpleAgent] = None

@app.on_event("startup")
async def startup_event():
    """Initialize the agent on startup."""
    global agent
    try:
        langchain_url = os.getenv("LANGCHAIN_URL", "http://localhost:10005")
        adk_url = os.getenv("ADK_URL", "http://localhost:10002")
        
        logger.info(f"Initializing agent with LangChain: {langchain_url}, ADK: {adk_url}")
        
        agent = SimpleAgent(langchain_url, adk_url)
        await agent.initialize()
        
        
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global agent
    if agent:
        await agent.cleanup()
        logger.info("Agent cleanup completed")

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
        status['llm_status'] = 'available'
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
        
        response = await agent.process_query(request.message)
        
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

@app.get("/conversation")
async def get_conversation_history():
    global agent
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        history = await agent.get_conversation_history()
        return {"conversation_history": history}
    except Exception as e:
        logging.error(f"Error getting conversation history: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting conversation history: {str(e)}")

@app.delete("/conversation")
async def clear_conversation_history():
    global agent
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        success = await agent.clear_conversation_history()
        if success:
            return {"message": "Conversation history cleared successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to clear conversation history")
    except Exception as e:
        logging.error(f"Error clearing conversation history: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing conversation history: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
