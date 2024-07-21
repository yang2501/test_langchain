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

from plotly.graph_objs._figure import Figure

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
    response_type: str = Field(description="can hold string values: clarification_question, pandasai_prompt, custom_plotting_function, custom_anomaly_detection_function")
    response_content: str = Field(description="the response content being the actual clarification question OR prompt to pandasai OR custom plotting function header with all parameters OR custom anomaly detection function with all parameters")

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
    1. The bland_altman_plot function takes ONLY these parameters: df, endpoint1, endpoint2, bySeverityCategory=False
    2. The change_from_baseline_plot function takes ONLY these parameters: df, endpoint
    3. The original_plot_endpoint_distribution function takes ONLY these parameters: df, endpoint1, endpoint2, bySeverityCategory=False
    4. The plot_correlation function takes ONLY these parameters: df, endpoint1, endpoint2, bySeverityCategory=False
    5. The severity_category_confusion_matrix function takes ONLY these parameters: df, endpoint, visit1='Screening', visit2=None
    6. The categorized_strip_plot function takes ONLY these parameters: df, endpoint, gold_standard_endpoint, visit=None
    7. The two_endpoints_visualization_report function takes ONLY these parameters: df, endpoint1, endpoint2, gold_standard_endpoint, bySeverityCategory=False
    
        For functions that need 2 endpoints, if the user only gives one endpoint, ask for the other endpoint. 
    """

    system_content = (
        f"You are a helpful assistant specialized in identifying the user's intent and generating clarifying questions."
        f"The dataframe is ALREADY EXISTS IN LOCAL SCOPE."
        f"ASK AS FEW CLARIFYING QUESTIONS AS POSSIBLE."
        f"When a user's prompt is too vague, create a context-specific clarifying question using the parameters in the function list. "
        f"For instance, if the prompt likely requires calling a custom data visualization function, ask the user to specify the necessary parameters. "
        f"Below is the list of custom plotting functions and their parameters: {function_headers}. "
        f"Below is more information about the data in the dataframe. This is useful for matching the prompt to a function and its parameters"
        f"The unique endpoints in the data frame are: {unique_endpoints}. "
        f"The unique devices in the data frame are: {unique_devices}. The unique visits in the data frame are: {unique_visits}. "
        f"If no clarifying questions are needed, output the custom function header OR Pandas.AI natural language prompt directly."
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
    elif response["response_type"] == "pandasai_prompt":
        # Add skills to the Agent
        vectorstore = ChromaDB()
        description = "You're a data analyst. When user asks you to plot or draw the graph of anything, you should provide the code to visualize the answer using plotly."
        agent = Agent(df, config={"llm": langchain_llm, "response_parser": StreamlitResponse}, vectorstore=vectorstore, description = description)
        return agent.chat(response["response_content"])
    elif response["response_type"] == "custom_plotting_function":
        st.write("function header is:")
        st.write(response["response_content"])
        
        from add_skills import bland_altman_plot, change_from_baseline_plot, plot_endpoint_distribution, plot_correlation, severity_category_confusion_matrix, categorized_strip_plot, two_endpoints_visualization_report
        # Prepare the local scope with necessary functions and data
        local_scope = {
            'df': df,
            'bland_altman_plot': bland_altman_plot,
            'change_from_baseline_plot': change_from_baseline_plot,
            'plot_endpoint_distribution': plot_endpoint_distribution,
            'plot_correlation': plot_correlation,
            'severity_category_confusion_matrix': severity_category_confusion_matrix,
            'categorized_strip_plot': categorized_strip_plot,
            'two_endpoints_visualization_report': two_endpoints_visualization_report
        }
        code = f"""
import pandas as pd

result = {response["response_content"]}

"""
        exec(code, {}, local_scope)
        result = local_scope['result']
        return result 

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if all(isinstance(fig, Figure) for fig in message["content"]):
            for fig in message["content"]:
                st.plotly_chart(fig)
        else:
            st.markdown(message["content"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.write(df.head(5))
    
    # Display the prompt text area
    if prompt := st.chat_input("Enter your prompt:"):

        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("assistant"):
            response = call_langchain_agent(df, langchain_llm, prompt)
            from plotly.graph_objs._figure import Figure
                
            if all(isinstance(fig, Figure) for fig in response):
                for fig in response:
                    st.plotly_chart(fig)
            else:
                st.write(response)
                    
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
            

    
#Give me a visualization report comparing ahi and pahi. ahi is the gold standard endpoint and plot by severity category. use the uploaded dataframe. you don't need more information