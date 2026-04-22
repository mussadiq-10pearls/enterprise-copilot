# test_llm.py
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

load_dotenv()

try:
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT"), # Explicitly set model
        api_version="2024-02-15-preview",
        temperature=0,
    )
    response = llm.invoke("Hello")
    print("Success:", response.content)
except Exception as e:
    print("Error:", e)