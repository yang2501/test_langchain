from azure.identity import ClientSecretCredential
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv
import os
from pandasai.smart_dataframe import SmartDataframe
from pandasai import Agent
import pandas as pd
from langchain_openai.chat_models import AzureChatOpenAI
from setup_azure_openai import get_llm

llm = get_llm()

# Configure PandasAI
# see config section at "https://docs.pandas-ai.com/getting-started" for available options
config = {
    "llm": llm,
    # Other configuration options as needed
}

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from pandasai.ee.vectorstores import ChromaDB


def create_agent(df, prompt):
    from pandasai import Agent
    from add_skills import add_skills_to_agent
    from qa_train import train_agent

    # Instantiate the vector store
    vector_store = ChromaDB()
    description = "You're a data engineer. You're very good at processing and transforming data before calling the custom skills."
    config={"llm": llm}

    description = "You're a data analyst."
    agent = Agent(df, config=config)
    # agent = Agent(df, vectorstore=vector_store, config=config, description=description)
    # add_skills_to_agent(agent=agent)
    # train_agent(agent=agent)
    agent.chat(prompt)
    return agent

    
    