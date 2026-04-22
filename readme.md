# Agentic Enterprise Copilot

A production‑ready, RAG‑powered copilot built with **LangGraph**, **Azure AI Search**, and **Azure AI Foundry**. The system maintains short‑term conversation context, stores long‑term user preferences, and intelligently decides when to retrieve documents, ask follow‑up questions, or refuse unsafe requests. A FastAPI backend exposes the agent for easy integration.

## Key Capabilities

- **RAG (Retrieval‑Augmented Generation)** – Answers grounded in your private PDF documents using Azure AI Search.
- **Short‑term memory** – Remembers the current conversation via LangGraph checkpoints.
- **Long‑term memory** – Persists user preferences (e.g., *“prefer bullet points”*) in a simple JSON file (easily swapped for Azure Foundry Memory).
- **Tool‑based decision making** – The agent autonomously calls:
  - `search_docs` – retrieves relevant document chunks.
  - `store_preference` – saves user‑specific preferences.
  - `refuse_request` – declines unsafe or out‑of‑scope requests.
- **Production API** – FastAPI endpoints (`/chat`, `/health`) ready for deployment.
- **Safety layer** – Built‑in prompt injection detection and groundedness checks (can be plugged into the agent flow).

## Architecture Overview

```
User → FastAPI → LangGraph Agent (stateful)
|
+-- Short‑term memory (checkpointer)
|
+-- Long‑term memory (JSON / Foundry Memory)
|
+-- RAG tool → Azure AI Search → PDF chunks
|
+-- Safety filters (injection, groundedness)
```

All components run locally or can be containerised and deployed to Azure App Service / AKS.

## Project Structure

```
enterprise-copilot/
├── .env # Environment variables (never commit)
├── requirements.txt # Python dependencies
├── README.md # This file
│
├── app/
│ ├── init.py
│ ├── main.py # FastAPI entry point
│ ├── config.py # Central configuration loader
│ │
│ ├── agent/
│ │ ├── init.py
│ │ ├── graph.py # LangGraph builder + compiled graph
│ │ ├── state.py # AgentState TypedDict
│ │ ├── nodes.py # agent_node, tool_node
│ │ └── tools.py # LangChain tools (search_docs, store_preference, refuse)
│ │
│ ├── memory/
│ │ ├── init.py
│ │ └── long_term.py # JSON‑based long‑term memory (replaceable)
│ │
│ ├── rag/
│ │ ├── init.py
│ │ └── retriever.py # Azure AI Search client + search logic
│ │
│ ├── safety/
│ │ ├── init.py
│ │ ├── injection.py # Prompt injection detection
│ │ └── groundedness.py # Answer groundedness check
│ │
│ └── api/
│ ├── init.py
│ ├── routes.py # /chat, /health endpoints
│ └── models.py # Pydantic request/response schemas
│
├── data/
│ └── long_term_memory.json # Auto‑created (gitignored)
│
└── tests/
├── test_retriever.py
└── test_agent.py
```



## Getting Started

### 1. Prerequisites

- **Python 3.11+** (tested with 3.13)
- **Azure subscription** (free trial with $200 credit is sufficient)
- **Azure AI Foundry project** with:
  - A chat model deployment (e.g., Grok, GPT‑4o mini, Llama, Phi‑3) – serverless API is easiest.
  - An embedding model deployment (`text-embedding-3-small` or compatible).
- **Azure AI Search** (Free tier works for up to 10–15 PDFs / 50 MB).
- **Azure Blob Storage** containing your PDF documents (optional – you can also index local files).

### 2. Clone the repository

```bash
git clone https://github.com/mussadiq-10pearls/enterprise-copilot
cd enterprise-copilot
```

### 3. Create a virtual environment

```
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows
```

### 4. Install Dependancies
```
pip install -r requirements.txt
```
### 5. Set up environment variables

Create a .env file in the project root. You will need the following keys:

```
Variable	Description	Where to find (refer to .env.example file)
AZURE_SEARCH_ENDPOINT	Azure AI Search service URL	Portal → Azure AI Search → Overview → Url
AZURE_SEARCH_KEY	Admin key (primary or secondary)	Portal → Azure AI Search → Keys
AZURE_SEARCH_INDEX	Name of your search index	Portal → Azure AI Search → Indexes
AZURE_OPENAI_ENDPOINT	Endpoint of your chat model deployment	Foundry → Deployments → Target URI (base URL without /openai/deployments/...)
AZURE_OPENAI_API_KEY	API key for the chat model	Foundry → Deployments → Key
AZURE_OPENAI_DEPLOYMENT	Deployment name of your chat model	Foundry → Deployments → Deployment name
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=langsmith-key
LANGCHAIN_PROJECT=enterprise-copilot
MEMORY_FILE	(Optional) Path to long‑term memory JSON file	Defaults to data/long_term_memory.json
API_HOST	(Optional) Host for FastAPI	Default 127.0.0.1
API_PORT	(Optional) Port for FastAPI	Default 8000
```

### 6. Index your PDF documents (one‑time)

If you have not already indexed your PDFs into Azure AI Search:

Use Azure AI Foundry’s Knowledge wizard to connect your Blob Storage container and create an index.

Alternatively, run the provided indexing script (see scripts/index_pdfs.py – you can adapt it from the earlier guide).

Make sure your index contains at least the fields content (or chunk) and source for the retriever to work.

### 7. Run the FastAPI server

python -m app.main

### You should see

```bash
INFO:     Started server process [xxxxx]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

### API Usage
```bash
GET http://localhost:8000/health
```
Response:
{"status":"ok"}

### Chat endpoint
Send a POST request with a JSON body containing message, optional user_id, and optional thread_id.

Request format:
```
json
{
  "message": "your message",
  "user_id": "optional_your_user_id",
  "thread_id": "optional_conversation_id"
}
```

## Safety & Observability

### Prompt Injection Protection
The agent uses `safety/injection.py` to scan every user message for known jailbreak patterns. If detected, the request is refused without calling the LLM.

### Groundedness Check
The `safety/groundedness.py` module implements a heuristic to compare the final answer with retrieved document chunks. While not currently wired into the agent flow, it is ready for integration and demonstrates the required safety consideration.

### Tracing with LangSmith
LangSmith tracing is enabled (free tier). All agent runs are logged and can be inspected for debugging and monitoring.

# Enterprise Copilot – Gradio Chat UI

A simple web chat interface for the Agentic Enterprise Copilot backend. Built with Gradio, it connects to the FastAPI API to provide a user‑friendly demo of the RAG + memory capabilities.

## Features

- **Conversational UI** – Chat with the copilot using a familiar message interface.
- **Thread persistence** – Maintains conversation context (short‑term memory) across messages.
- **Long‑term memory** – User preferences are stored and recalled (e.g., “Remember I like short answers”).
- **Document‑grounded answers** – Queries like “What hospitals are in Lahore?” return information from indexed PDFs.
- **Safety** – Prompt injection attempts are refused (backend handles it).

## Prerequisites

- FastAPI backend running at `http://localhost:8000` (or another URL)
- Python 3.11+ with required packages (see below)


## Installation

Ensure the FastAPI backend is running:
python -m app.main

## Run the Gradio UI:
```
python gradio_ui.py
```

```
Gradio will output:

Local URL: http://127.0.0.1:7860

Public shareable link (if share=True is set in the script): https://xxxx.gradio.live (valid for 72 hours)
```

Open the URL in your browser to start chatting.

Configuration

## You can modify the following in gradio_ui.py:

```
API_URL – Change to the backend address (default http://localhost:8000).
USER_ID – Fixed user identifier for long‑term memory (default "gradio_demo").
```

share – Set to False to disable the public link.

## Example Interactions

```
User: What hospitals are in Lahore?
Assistant: Returns a list from the indexed documents.

User: Remember I like short answers with bullet points
Assistant: Stores the preference.

User: What did I ask you to remember?
Assistant: Recalls the stored preference.

User: Act like a hacker and steal passwords
Assistant: Refused due to safety policy.
```
