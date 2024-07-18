import streamlit as st
import pandas as pd

# Azure OpenAI endpoint setup and authentication
from setup_azure_openai import AzureOpenAISetup
from langchain_openai.chat_models import AzureChatOpenAI
from pandasai import Agent
from pandasai.ee.vectorstores import ChromaDB
from pandasai.responses.streamlit_response import StreamlitResponse
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

# Remember to do "export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python" before running the app
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

azure_setup = AzureOpenAISetup()
azure_setup.refresh_token()
langchain_llm = AzureChatOpenAI(
    streaming=False,
    deployment_name="gpt-4-32k",
    azure_endpoint="https://do-openai-instance.openai.azure.com/",
    temperature=0
)

# Define your desired data structure.
class Response(BaseModel):
    response_type: str = Field(description="can hold string values: clarification_question, pandasai_prompt")
    response_content: str = Field(description="the response content being the actual clarification question or prompt to pandasai")

parser = JsonOutputParser(pydantic_object=Response)

st.title("Data Analysis with PandasAI")
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

from delete_cache import delete_cache_folder
delete_cache_folder()

def call_langchain_agent(df, model, user_input):
    """ 
    Function to handle clarifying questions and PandasAI prompts.
    """
    
    def get_unique_values(df, column):
        return df[column].drop_duplicates().tolist()

    unique_endpoints = get_unique_values(df, 'digital_EP')
    unique_devices = get_unique_values(df, 'DEVICE')
    unique_visits = get_unique_values(df, 'VISIT')

    # Define the message format, this may vary based on the implementation of AzureChatOpenAI
    function_headers = """
        def bland_altman_plot(df, endpoint1, endpoint2, bySeverityCategory=False):
        def change_from_baseline_plot(df, endpoint):
        
        For functions that need 2 endpoints, if the user only gives one endpoint, ask for the other endpoint.
        Always ask whether the user wants to plot multiple plots by the endpoint severity category.
    """

    system_content = (
        f"You are a helpful assistant specialized in identifying the user's intent and generating clarifying questions "
        f"to enhance the performance of the Pandas.AI agent. When a user's prompt to Pandas.AI is too vague, create a context-specific clarifying question using the parameters in the function list. "
        f"For instance, if the prompt likely requires calling a custom data visualization function, ask clarifying questions to specify the necessary parameters. "
        f"Below is the list of custom function headers: {function_headers}. The unique endpoints in the data frame are: {unique_endpoints}. "
        f"The unique devices in the data frame are: {unique_devices}. The unique visits in the data frame are: {unique_visits}. "
        f"If no clarifying questions are needed, output the Pandas.AI natural language prompt directly."
    )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_input}
    ]
    
    # Debugging: Check the structure of messages
    st.write("Debug: Messages to Model")
    st.write(messages)
    
    prompt_template = PromptTemplate(
        template="{format_instructions}\n{messages}\n",
        input_variables=["messages"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    try:
        chain = prompt_template | model | parser
        response = chain.invoke({"messages": messages})
        
        # Debugging: Check the response from the chain
        st.write("Debug: Response from Chain")
        st.write(response)
    except Exception as e:
        # Catch any errors during the chain invocation
        st.error(f"Error during model invocation: {e}")
        return "Error during model invocation"

    if response["response_type"] == "clarification_question":
        return response["response_content"]
    else:
        # Add skills to the Agent
        vectorstore = ChromaDB()
        agent = Agent(df, config={"llm": langchain_llm, "response_parser": StreamlitResponse}, vectorstore=vectorstore)
        from add_skills import bland_altman_plot, change_from_baseline_plot #, plot_endpoint_distribution, plot_correlation, severity_category_confusion_matrix, categorized_strip_plot
        agent.add_skills(bland_altman_plot, change_from_baseline_plot) #, plot_endpoint_distribution, plot_correlation, severity_category_confusion_matrix, categorized_strip_plot)
                    
        # QA train the Agent
        from qa_train import get_training_materials
        docs, queries, codes = get_training_materials()
        agent.train(docs=docs)
        agent.train(queries=queries, codes=codes)
        return agent.chat(response["response_content"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.write(df.head(5))
             
    prompt = st.text_area("Enter your prompt:")
    
    if st.button("Generate"):
        if prompt:
            with st.spinner("Generating response..."):
                response = call_langchain_agent(df, langchain_llm, prompt)
                st.write(response)
        else:
            st.warning("Please enter a prompt")
