# config.py
import os
from dotenv import load_dotenv
import httpx
from langchain_openai import AzureChatOpenAI

# Optional token cache directory
tiktoken_cache_dir = "./token"
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir

load_dotenv()  # loads .env variables

api_key        = os.getenv("AZURE_OPENAI_API_KEY")
endpoint       = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version    = os.getenv("AZURE_OPENAI_API_VERSION")
deployment     = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# Shared HTTP client (disables SSL verification – keep False only for dev)
client = httpx.Client(verify=False)

# Central LLM instance for the whole app
llm = AzureChatOpenAI(
    azure_endpoint=endpoint,
    openai_api_key=api_key,
    openai_api_version=api_version,
    azure_deployment=deployment,
    http_client=client,
    temperature=0.3,
)
