import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

load_dotenv()

endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
key = os.getenv("AZURE_SEARCH_KEY")
index_name = os.getenv("AZURE_SEARCH_INDEX")  # use the variable from .env

print(f"Endpoint: {endpoint}")
print(f"Index: {index_name}")

credential = AzureKeyCredential(key)
search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)

try:
    results = search_client.search(search_text="*", top=1, include_total_count=True)
    print("Search successful!")
    for result in results:
        print(result)
except Exception as e:
    print(f"Error: {e}")
    if hasattr(e, 'response'):
        print("Response body:")
        print(e.response.text)