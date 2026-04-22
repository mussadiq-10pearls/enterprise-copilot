from langchain.tools import tool
from app.rag.retriever import search_company_docs as search_company_docs_raw
from app.memory.long_term import save_memory

@tool
def search_docs(query: str) -> str:
    """Search internal documents for information."""
    result_str, _ = search_company_docs_raw(query)
    return result_str

@tool
def store_preference(user_id: str, key: str, value: str) -> str:
    """Store a user preference."""
    save_memory(user_id, key, value)
    return f"Stored {key} = {value}"

@tool
def refuse_request(reason: str) -> str:
    """Refuse unsafe or off-topic requests."""
    return f"REFUSED: {reason}"