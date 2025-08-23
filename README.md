# Real Estate Investment Analysis System Overview
A comprehensive AI-powered real estate investment analysis system that combines multiple AI agents, RAG pipelines, and advanced search capabilities to provide intelligent property insights and investment recommendations. It is developed to help buyers find their suitable properties in US and get an advise whether it is a good investment or not based on tax rate, property policies and geo-location of the property (neiborhood and nearby amenities, in addition to that it can calculate distanse from the listed property you liked with the address of your work place, in order to check if it suits you or not)

## Project Checklist
- [x] Set up MCP server (adk_mcp_server.py and search_mcp_server.py)
- [x] Set up adk agent for tax analysis, Google maps and Gmail services (root_agent.py)
- [x] Set up langchain agent for param extraction and property search using the RentCast API (property_search.py)
- [x] Used RAG pipelines for handeling CSV and PDF files (rag_pipeline_csv and rag_pipeline_pdf)
- [x] Used an orchestrator to handle routing between agents (simple_agent.py)
- [x] Defined a fastApi server that connects the orchestrator with the frontend
- [x] Fine-tuned a Qwen-2.5 model on the HouseTS dataset to function as an advisor (Qwen-Fine_Tuning.ipynb)


## Architecture

### Core Components

#### 1. **MCP Tools for Langchain Agent** (`app/mcp_tools/search_mcp_server.py`)
- As mentioned before this agent is used to extract parameters from the user's query and search for those parameters on RentCast
- Those tools were deployed on an mcp server (search_mcp_server.py)
- One challenge was to properly handle the parameters extraction, as RentCast is accurate with the searching to be used when searching (https://developers.rentcast.io/reference/sale-listings). For that, Gemini-2.0 was used with a proper set of instruction to help the LLM parse those parameters(extraction_instructions.txt).
- I faced a prompt engineering issue here, because the LLM had to be instructed carefully on how to handle different context and queries and result with a proper response
- Additionally, the method 'ast.literal_eval' was used to ensure the result of this tool is a dictionary.
- Now for property search, I used the api request mentioned and an LLM to use the json response and format it in a proper and professional respond.

#### 2. **Langchain Agent System** (`app/langchain_agent/`)
- This agent was initially built using LangChain's traditional patterns to orchestrate multiple tools (extract_param, search_properties).
- One major challenge was state persistence in LangChain. The outputs from extract_param were not being properly carried over to the search_properties tool. This caused the chain to break or pass None values, making the workflow unreliable and breaking the tool chain
- Another issue was with the schema expectations in LangChain. The framework expects inputs in a structured format like {"input": "...query..."}, while sometimes the tools returned raw strings or differently structured dictionaries. This mismatch led to validation errors and broken flows that required constant debugging and schema alignment.
- I also struggled with tool routing in LangChain. The agent was ignoring available tools and trying to answer directly with the LLM instead of invoking search_properties. I solved this by trying stricter prompt engineering with clearer instructions.

- Solution: After a long experimentation with LangChain's patterns, I shifted away from LangChain's agent framework and built a custom StateGraph workflow using LangGraph. This approach eliminated state persistence issues by implementing custom conversation state management with conversation_states dictionary, bypassed schema validation problems by directly integrating MCP tools without LangChain's tool abstractions, solved tool routing issues by creating explicit workflow nodes (extract_param_node, search_properties_node) with direct tool invocation, replaced LangChain patterns with a custom StateGraph that maintains state through explicit node transitions, and integrated directly with A2A protocol for robust agent-to-agent communication which will talk about it later.


#### 3. **MCP Tools for ADK Agent** (app/mcp_tools/adk_mcp_server.py)
- This MCP server provides specialized tools for the ADK real estate agent, including document analysis, email communication, location services, and distance calculations
- The tools were deployed on an MCP server (adk_mcp_server.py) running on port 8001, which serves as the backend for the ADK agent's specialized capabilities
- One challenge was implementing the RAG pipeline tool for document analysis. The tool needed to handle various types of real estate documents and queries while maintaining context and providing accurate, relevant responses. For this, Gemini-2.0 was used with proper prompt engineering to ensure the RAG system could retrieve and synthesize information effectively from the document database
- I faced integration challenges with the Gmail API tool for creating email drafts. The tool needed to handle different recipient formats, agent names, and property addresses while ensuring proper error handling and successful draft creation
- For the distance calculation tool, I had to properly integrate with Google Maps API to handle different travel modes (driving, walking, bicycling) and ensure accurate distance and duration calculations between various location formats.
- The nearby places search tool required careful handling of different place types and address formats, with proper error handling to ensure the tool could handle API failures or invalid inputs


#### 4. **ADK Agent System** (`app/adk_agent/`)
- I implemented the ADK agent system with a root agent (real_estate_root_agent) and several specialized sub-agents:
    - document_analysis_agent: Analyzes real estate documents using the RAG pipeline, focusing on tax info, compliance, and market analysis.
    - communication_agent: Manages professional email drafting and client communication using Gmail tools.
    - nearby_places_agent: Retrieves and analyzes nearby amenities (schools, hospitals, restaurants, etc.) for neighborhood insights.
    - distance_calculation_agent: Calculates commute distances and travel times for accessibility comparisons.
    - real_estate_root_agent: Acts as the main entry point, capable of invoking all specialized agents and tools directly through the main MCP toolset
- The main issue faced was configuring the MCP connection parameters.
- At first I used a plain dictionary with command and args, This approach caused validation errors in ADK because the framework expected an explicit connection class
- The solution was to use seConnectionParams, which properly defines the MCP server URL and timeout.



#### 4. **RAG Pipelines** (`app/mcp_tools/`)
- I used 2 pipelines because I have CSV and PDF data, which should be handled and processed differently
- The same vector DB was used to store both information:
    - for my use case i used Chroma DB and not FAISS
    - Chroma db helped me while testing the RAG in tracking from which document it provided the information
    - FAISS stores vectors without context, meaning where we lose track of which document each chunk came from.
    - I have also worked with Chroma db before for that I used here in this project
- for pdfs:
    - PDFs are long, unstructured documents for that I used UnstructuredPDFLoader()
    - 1000 character chunks maintain context
    - 100 character overlap prevents information loss
    - Recursive splitting handles various document formats
    - from prevoius experience, I had an issue with storing duplicate data in the db, for that I used this
    - I used the BAAI/bge-m3 from HuggingFaceEmbeddings to embedd my data. 
    function calculate_chunk_ids() which ensures the RAG system doesn't waste resources storing duplicate information. Ref: Ref: https://www.youtube.com/watch?v=2TJxpyO3ei4&t=946s

- for csv:
    - same context as pdfs but I didn't split the csv files, as they are already structured and the columns are up to 4-5 columns only, only used CSVLoader().

- Note: you can find in app\mcp_tools\config.py that I'm following the singleton pattern. Because at first I was defining the same models (HuggingFace embeddings, ChromaDB, and Gemini LLM) in multiple files like rag_pipeline_pdf.py, rag_pipeline_csv.py, and adk_mcp_server.py, which caused the BAAI/bge-m3 embedding model to download multiple times (once for each file), created multiple ChromaDB instances that conflicted with each other when trying to access the same vector database files, and resulted in wasted memory from duplicate model instances and slower startup times as each component waited for its own model initialization. following this way, the pattern ensured that each model is created only once when first requested and then reused across all subsequent calls.

#### 5. **A2A protocol Adk** (`app/adk_agent/`)

- **A2A Architecture Overview:** 
- same logic as for langchain but using adk here
    - Main Entry Point (__main__.py):
        - Starts ADK server on port 10002
        - Creates agent card for discovery by other agents
        - Defines capabilities (streaming, push notifications)
        - Sets up HTTP handler with the ADK agent executor
    
    - Agent Executor (agent_executor.py):
        - Bridge between ADK and the root agent
        - Handles ADK protocol (requests, responses, streaming)
        - Manages task lifecycle (start, update, complete)
        - Converts ADK format to the agent's format

    - ADK Agent (root_agent.py):
        - Handles business logic (document analysis, communication, location services, distance calculations)
        - Manages conversation state per context_id
        - Uses sub-agent orchestration for specialized task handling
        - Streams responses back to ADK executor
        what's new for adk is:
        - Root Agent Class Features:
            - Ready for Google ADK communication with sub-agent orchestration
            - maintains separate conversation states for different users through context_id
            - Routes queries to specialized agents (document_analysis_agent, communication_agent, nearby_places_agent, distance_calculation_agent)
            - Each sub-agent connects to MCP tools server for specialized functionality
            - Coordinates complex multi-agent workflows through sub-agent delegation

- Note: This work flow was referenced from: https://github.com/bhancockio/agent2agent/tree/main/a2a_friend_scheduling/karley_agent_adk


#### 6. **A2A protocol Langchain** (`app/langchain_agent/`)

- **A2A Architecture Overview:** 
    - Main Entry Point (__main__.py):
        - Starts A2A server on port 10005
        - Creates agent card for discovery by other agents
        - Defines capabilities (streaming, push notifications)
        - Sets up HTTP handler with the agent executor
    
    - Agent Executor (agent_executor.py):
        - Bridge between A2A and the root agent
        - Handles A2A protocol (requests, responses, streaming)
        - Manages task lifecycle (start, update, complete)
        - Converts A2A format to the agent's format

    - Langchain Agent (property_search.py):
        - Handles business logic (extract params and property search)
        - Manages conversation state per context_id
        - Uses LangGraph workflow for tool orchestration
        - Streams responses back to A2A executor
        - RealEstateAgent Class Features:
            - defined for Agent-to-Agent communication with streaming responses
            - maintains separate conversation states for different users through context_id
            - connected to MCP tools server.
            - Orchestrates extract_params -> search_properties workflow through StateGraph
            - handles retry logic, timeout protection, and graceful error recovery
            - Provides real-time progress updates and smart clarification requests

- Note: This work flow was referenced from: https://github.com/bhancockio/agent2agent/tree/main/a2a_friend_scheduling/kaitlynn_agent_langgraph/app



#### 7. **Orchestrator System** (`app/orchestrator_agent/`)
- **Orchestrator  Architecture Overview:** 
    - Entry Point Orchestrators (Used by Simple Agent):
        - ADK Orchestrator (adk_orchestrator.py):
            - Entry point for connecting to ADK agents via A2A protocol
            - Communicates only with root_agent for sub-agent orchestration
            - Handles ADK-specific response parsing and content extraction
            - Manages streaming responses with 4-minute timeout for complex workflows
            - Routes all queries through the root agent, which internally delegates to specialized sub-agents

        - LangChain Orchestrator (orch_agent.py):
            - Entry point for connecting to LangChain agents via A2A protocol
            - Communicates with real_estate agent for property search
            - Handles LangChain-specific response parsing and content extraction
            - Manages streaming responses with 3-minute timeout for property queries
            - Direct communication with LangChain agent for specialized real estate tasks

    - Simple Agent (simple_agent.py):
        - Coordinates between ADK and LangChain orchestrators
        - Uses Gemini 2.0 for response enhancement
        - saves conversation history and context:
            - In the __init__ method I defined conversation_history which stores a list of dictionaries containing conversation messages, Each message has: role, content, timestamp. This memory management keeps only last 10 messages
            
          - How it works:
            - I defined it to append each new message to the list.
            - tracks when each message was sent with the Timestamp
            - automatically removes oldest messages (keeps last 10)
            - Refer to _add_to_conversation_history method

          - What this provides:
            - Last 5 messages from conversation history
            - Formatted context for Gemini to understand conversation flow
            - Role identification (User vs Assistant) for context clarity
            - Refer to _get_conversation_context method

          - Context Integration with Gemini:
            - I passed the conversation context to the LLM by using a proper prompt, defining its job
            - Refer to _format_response_with_gemini method

- Of course there is a better way to handle conversation history, but I had to define it in this way becuase that phase came last and by that time I was researching on how to fine-tune an LLM which will be talked about later.

#### 8. **A2A Challenges and Solutions**

- Challenges:
    - A2A Framework Response Type Mismatch:
        - The A2A framework expected specific response types but our agent was returning incompatible formats
        - This caused "Agent returned invalid type response for this method" errors
        - Non-streaming requests failed completely because of this mismatch

    - MCP Server Connection Issues:
        - Sometimes the MCP client (simple agent) was failing to connect to search server, causing "generator didn't yield" errors
        - There were Timeout issues because the RentCast API was a bit slow (for extracting param, making a request, llm invking the response... as we mentioned earlier)

    - Queue Management Complexity:
        - A2A event queue methods were unclear and caused "Queue is closed" errors
        - I struggled with incorrect queue usage patterns
        - because of that the streaming responses couldn't be delivered to the orchestrator

    - I also had a lot of challenges regarding package version compatibilty, because different versions had different class names and function signatures

    - The httpx.AsyncClient() was being closed too early, causing "client has been closed" errors. This happened because A2A's InMemoryPushNotifier needs the client to stay open

    - Simple messages were timing out after 5 seconds because real estate searches take too long. This happened because A2A client has internal timeout limits for non-streaming requests ()

- Solutions:
    - I added timeout handling, retry logic. 
    - The implementation uses async context managers with timeout protection
    - This resulted in a stable connection to MCP search server

    - And in addition to timeouts, the main issue was the response format for that I solved by implementing conditional response logic in the agent_executor.py file that detects whether a request is streaming or non-streaming based on the presence of an event_queue parameter. For streaming requests, the agent returns an empty string (which A2A expects) and sends data through the event_queue using enqueue_event() for real-time updates, while for non-streaming requests, the agent returns the full response data structure that A2A expects. This fix ensures that A2A gets the correct response type for each request mode - minimal responses for streaming (where data flows through the queue) and complete data structures for non-streaming (where data flows through the return value) - eliminating the "Agent returned invalid type response for this method" errors and allowing both streaming and non-streaming functionality to work properly. However, despite implementing this conditional logic, the streaming functionality was still facing issues where responses were being delivered as non-streaming without progress updates, and unfortunately I couldn't solve this streaming issue completely due to time constraints and the complexity of the A2A framework's internal behavior.


#### 9. **FastAPI server and Frontend** (`app/orchestrator_agent/`)

- FastAPI Server Architecture (fastapi_server.py):
    - API Gateway: Acts as the main entry point for frontend communication
    - Agent Integration: Connects the Simple Agent to web interface through RESTful endpoints
    - Environment Configuration: Loads LangChain and ADK URLs from environment variables for flexible deployment
    - CORS Support: Enables cross-origin requests for web frontend integration
    - Health Monitoring: Provides status endpoints to monitor agent and orchestrator health

- Integration Flow:
    - Frontend -> FastAPI: HTML interface sends HTTP POST requests to /chat endpoint
    - FastAPI -> Simple Agent: Server routes requests to SimpleAgent.process_query() method
    - Simple Agent -> Orchestrators: Routes queries to appropriate LangChain or ADK orchestrator
    - Response Flow: Results flow back through Simple Agent -> FastAPI -> Frontend with proper formatting
    - Status Updates: Frontend polls /status endpoint to monitor system health and agent availability

- I tried to implement the frontend using Streamlit but faced incompatible package versions and dependency conflicts, for that I switched to using HTML/CSS/JavaScript frontend.


#### 10. **Fine-Tuning Model** (`Qwen_Fine_Tuning.ipynb`)
- I first started by searching for models, I checked Mitral 7B and openAI models, but they were either heavy or expensive, so i used Qwen 2.5 which consists of 3B parameters

- I used the HouseTS dataset which is governmental dataset with comprehensive real estate and demographic data.

- 1) Created a quantitative scoring system based on:
        - Market velocity (days on market, sale-to-list ratios)
        - Economic indicators (unemployment, income levels)
        - Supply/demand balance (inventory, new listings)
        - Note: Chatgpt helped me with this, by implementing a simple scoring algorithm
- This new shaped data was saved in a jsonl file "investment_training.jsonl"


- 2) The dataset I used (HouseTS) contains 25k records, so using it all would take several training days, so I just used 1,500 sample only
- I splitted this jsonl file into traiing and testing with 0.1 test size

- 3) Before tokenizing I formatted each property example and combine the question and answer into one text string
- ex: property = {
    'prompt': "Analyze this real estate market data: Austin, TX - $850k median price...",
    'completion': "STRONG BUY: Excellent investment opportunity..."
}
convert it into: "Analyze this real estate market data: Austin, TX - $850k median price...  STRONG BUY: Excellent investment opportunity..."

- then tokenized those combined those text

- 4) Used DataCollatorForLanguageModeling() which is a function that organizez the data into batches so the LLM can learn from
- Note: there is a parameter called mlm (masked Language model)
- mlm is designed for fill-in-the-blank, not generation; it masks some of the words and try to predict them from the context (Bert models use it)
- Qwen is Causal LM meaning it predicts next token left-to-right just as GPT-style architecture
- for that we define mlm as false

- 5) I defined the training settings such as batch-size, number of epochs, learning rate etc...

- 6) Defined the trainer;
        - Now because Qwen is a quentized model, meaning that its weights are stored in compressed/quantized format to save memory and speed up inference.
        - The solution was to use LoRa; which adds small "adapter layers" on top of the frozen quantized model
        - So now only the tiny LoRA layers get trained 

- 7) After configuring LoRa, I had to use another training settings:
        - LoRA only adds a tiny fraction of trainable parameters; so I used Batch size = 2, accumulation = 4( This improves gradient estimates and training stability.)
        - Also because of what LoRa does I used fp16 because LoRA only trains small adapter layers on top of the frozen quantized model. That makes fp16 safe to use, and it gives me faster training and lower GPU memory usage without hurting performance
            - what is fp16?
                - is a 16-bit floating-point number format that uses half the memory of fp32, making training faster and lighter but with lower numerical precision.
        - Changed the optimizer and used plain adamw which is defualt in TrainingArguments
            - why?
            - because before I was trying to train the full model so using paged_adamw_32bit would have helped as it is a memory-efficient optimizer
            - and because LoRA reduces trainable parameter count so much that I don’t need those extreme memory-saving tricks anymore.
        - also i increased the number of epochs into 3 as LoRa is lightweight, so longer training is affordable

- 8) Lastly I defined my Trainer another time and trained the model

- resulted with a 0.548 training loss

### ** Due to the project deadline I wasn't able to fit this LLM into my project, as I was looking to integrate it in the simple_agent and be like an advisor for the user whether he/she should buy a property or not **



### Core Technologies
- **Python 3.8+**
- **Google ADK (Agent Development Kit)**
- **LangChain & LangGraph**
- **Model Context Protocol (MCP)**
- **Chroma Vector Database**
- **FastAPI & Uvicorn**

### AI Models
- **Gemini 2.0 Flash** (Primary LLM)
- **HuggingFace Embeddings** (Document embeddings)
- **Sentence Transformers** (Text processing)


### Infrastructure
- **ChromaDB** (Vector storage)
- **SQLite** (Metadata storage)
- **Async HTTP** (Communication)



### Prerequisites
- Python 3.8 or higher
- Google Cloud credentials
- Real Estate API key
- LLM API key

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd Real-Estate-project
```

2. **Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
# Create .env file
REAL_ESTATE_API_KEY=your_api_key
LLM_API_KEY=your_llm_key
GOOGLE_APPLICATION_CREDENTIALS=path_to_credentials.json
```

5. **Initialize the system**
```bash
# Start MCP server
cd app/mcp_tools
python search_mcp_server.py

# Start the other mcp server (in another terminal)
cd app/mcp_tools
python adk_mcp_server.py

# Start orchestrator (in another terminal)
cd app/orchestrator_agent
python fastapi_server.py

# Start ADK agent (in another terminal)
cd app/adk_agent/real_estate_agent
python __main__.py

# Start langraph agent (in another terminal)
cd app/langchain_agent/
python __main__.py
```
## References:
- All reference links are stated in this markdown file under each phase, along with some links commented in the code files beside their usage.





