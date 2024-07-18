from azure.identity import ClientSecretCredential
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
import streamlit as st

from dotenv import load_dotenv
import os

class AzureOpenAISetup:
    def __init__(self):
        load_dotenv()
        self.tenant_id = st.secrets.openai.tenant_id # replaced os.environ.get("tenant_id")
        self.client_id = st.secrets.openai.client_id
        self.client_secret = st.secrets.openai.client_secret
        self.refresh_token()

    def refresh_token(self):
        credential = ClientSecretCredential(self.tenant_id, self.client_id, self.client_secret)
        self.token = credential.get_token("   https://cognitiveservices.azure.com/.default")
        os.environ["OPENAI_API_TYPE"] = "azure_ad"
        os.environ["OPENAI_API_KEY"] = self.token.token
        os.environ["AZURE_OPENAI_AD_TOKEN"] = self.token.token
        os.environ["OPENAI_API_VERSION"] = "2023-07-01-preview"

        self.embeddings = AzureOpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=self.token.token,
            azure_endpoint="https://do-openai-instance.openai.azure.com/",
        )

    def get_embeddings(self):
        return self.embeddings

    def get_token(self):
        return self.token

azure_setup = AzureOpenAISetup()
 # refresh token and update corresponding envs
# call this refresh_token if needed
azure_setup.refresh_token()


    
    
    