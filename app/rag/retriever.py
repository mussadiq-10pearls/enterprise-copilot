from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from app.config import SEARCH_ENDPOINT, SEARCH_KEY, SEARCH_INDEX

search_client = SearchClient(
    endpoint=SEARCH_ENDPOINT,
    index_name=SEARCH_INDEX,
    credential=AzureKeyCredential(SEARCH_KEY)
)

def search_company_docs(query: str):
    """
    Search Azure AI Search and return:
        (formatted_string, raw_chunks_list)
    """
    try:
        results = search_client.search(search_text=query, top=3)
        formatted = []
        raw_chunks = []
        for doc in results:
            # Adjust field names to match your index schema
            content = doc.get("content", doc.get("chunk", doc.get("text", str(doc))))
            source = doc.get("source", doc.get("filename", "Unknown"))
            formatted.append(f"[Source: {source}]\n{content[:500]}")
            raw_chunks.append(content[:500])
        if not formatted:
            return "No relevant documents found.", []
        return "\n\n---\n\n".join(formatted), raw_chunks
    except Exception as e:
        return f"Search error: {str(e)}", []