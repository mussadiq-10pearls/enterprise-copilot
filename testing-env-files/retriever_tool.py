import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from langchain.tools import tool

load_dotenv()

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX")

search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=INDEX_NAME,
    credential=AzureKeyCredential(AZURE_SEARCH_KEY)
)

def search_company_docs_func(query: str) -> str:
    """Search company documents using full-text search."""
    try:
        results = search_client.search(search_text=query, top=3)
        formatted = []
        for doc in results:
            content = doc.get("content", doc.get("chunk", doc.get("text", str(doc))))
            source = doc.get("source", doc.get("filename", "Unknown"))
            formatted.append(f"[Source: {source}]\n{content[:500]}")
        if not formatted:
            return "No relevant documents found."
        return "\n\n---\n\n".join(formatted)
    except Exception as e:
        return f"Search error: {str(e)}"

search_company_docs = tool(search_company_docs_func)