import streamlit as st
import pandas as pd

# Azure OpenAI endpoint setup and authentication
from setup_azure_openai import AzureOpenAISetup
from langchain_openai.chat_models import AzureChatOpenAI

# Plotly imports
from plotly.graph_objs._figure import Figure

# Local file imports
from history_maintanance import save_messages_to_file, load_messages_from_file
from call_pandasai import call_pandasai

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

st.title("Data Analysis with PandasAI")
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

from delete_cache import delete_cache_folder
delete_cache_folder()

def initialize_and_display_chat_history():
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = load_messages_from_file()
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
           # if all(isinstance(fig, Figure) for fig in message["content"]):
              #  for fig in message["content"]:
                  #  st.plotly_chart(fig)
          #  else:
            st.markdown(message["content"])

initialize_and_display_chat_history()

# Define the constant
MAX_CLARIFYING_QUESTIONS = 3

# Initialize number of clarifyin questions and man clarifying questions
if "num_clarifying_questions_asked" not in st.session_state:
    st.session_state.num_clarifying_questions_asked = 0
if "max_clarifying_questions" not in st.session_state:
    st.session_state.max_clarifying_questions = MAX_CLARIFYING_QUESTIONS
    
    
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.write(df.head(5))
    
    # Display the prompt text area
    if prompt := st.chat_input("Enter your prompt:"):

        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        save_messages_to_file(st.session_state.messages)
        
        with st.chat_message("assistant"):
            response, code = call_pandasai(df, langchain_llm, st.session_state.messages, st.session_state.num_clarifying_questions_asked, st.session_state.max_clarifying_questions)
                                                                                    
            
           # st.write("Number of clarifying questions asked")
           # st.write(st.session_state.num_clarifying_questions_asked)
            st.write(response)
            st.code(code, language="python", line_numbers=False)
                    
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        save_messages_to_file(st.session_state.messages)
    
# Give me a visualization report comparing ahi and pahi. ahi is the gold standard endpoint and plot by severity category. use the uploaded dataframe. you don't need more information