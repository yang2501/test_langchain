import streamlit as st

# Miscellaneous
import re
from enum import Enum

# Pandasai imports
from pandasai import Agent
from pandasai.ee.vectorstores import ChromaDB
from pandasai.responses.streamlit_response import StreamlitResponse

# Langchain imports
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator, root_validator

# Local file imports
from system_messages import get_system_message, get_summarized_session_state_messages

# Remember to do "export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python" before running the app
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Define Enums for Response Types
class ResponseType(str, Enum):
    CLARIFICATION_QUESTION = "clarification_question"
    PANDASAI_PROMPT = "pandasai_prompt"
    CUSTOM_PLOTTING_FUNCTION = "custom_plotting_function"
    CUSTOM_ANOMALY_DETECTION_FUNCTION = "custom_anomaly_detection_function"

# List of custom plotting function names
CUSTOM_PLOTTING_FUNCTIONS = [
    "bland_altman_plot", "change_from_baseline_plot", "plot_endpoint_distribution",
    "plot_correlation", "severity_category_confusion_matrix", "categorized_strip_plot",
    "two_endpoints_visualization_report"
]

# Define the Response Model with Custom Validation
class Response(BaseModel):
    response_type: ResponseType = Field(description="can hold enum values: clarification_question, pandasai_prompt, custom_plotting_function, custom_anomaly_detection_function")
    response_content: str = Field(description="the response content being: the actual clarification question OR prompt to pandasai with FULL CONTEXT OR custom plotting function header with all parameters OR custom anomaly detection function with all parameters")

    @validator("response_type", allow_reuse=True)
    def set_response_type(cls, value, values):
        response_content = values.get("response_content", "")
        if any(re.search(r'\b' + re.escape(func) + r'\b', response_content) for func in CUSTOM_PLOTTING_FUNCTIONS):
            return ResponseType.CUSTOM_PLOTTING_FUNCTION
        return value

    @validator("response_content", allow_reuse=True)
    def check_response_content(cls, value, values):
        response_type = values.get("response_type")
        if response_type == ResponseType.CUSTOM_PLOTTING_FUNCTION:
            if not any(re.search(r'\b' + re.escape(func) + r'\b', value) for func in CUSTOM_PLOTTING_FUNCTIONS):
                raise ValueError("Invalid custom plotting function content")
        return value
    
    @root_validator(pre=True, allow_reuse=True)
    def validate_parameters(cls, values):
        response_type = values.get("response_type")
        response_content = values.get("response_content")
        
        def get_unique_values(df, column_name):
            return df[column_name].unique().tolist()

        if response_type == ResponseType.CUSTOM_PLOTTING_FUNCTION:
            df = values.get("df")  # Assume df is passed as a part of the values for validation
            unique_endpoints = get_unique_values(df, 'digital_EP')
            unique_devices = get_unique_values(df, 'DEVICE')
            unique_visits = get_unique_values(df, 'VISIT')
            allowed_values = set(unique_endpoints + unique_devices + unique_visits + [True, False])

            # Extract parameters
            params = re.findall(r'\b\w+\b', response_content)
            for param in params:
                if param not in allowed_values:
                    raise ValueError(f"Invalid parameter: {param}")
        return values


class Max_Questions_Exceeded_Response(BaseModel):
    response_type: ResponseType = Field(description="can hold enum values: pandasai_prompt, custom_plotting_function, custom_anomaly_detection_function")
    response_content: str = Field(description="the response content being: the prompt to pandasai with FULL CONTEXT OR custom plotting function header with all parameters OR custom anomaly detection function with all parameters")

    @validator("response_type", allow_reuse = True)
    def set_response_type(cls, value, values):
        response_content = values.get("response_content", "")
        if any(re.search(r'\b' + re.escape(func) + r'\b', response_content) for func in CUSTOM_PLOTTING_FUNCTIONS):
            return ResponseType.CUSTOM_PLOTTING_FUNCTION
        return value

    @validator("response_content", allow_reuse = True)
    def check_response_content(cls, value, values):
        response_type = values.get("response_type")
        if response_type == ResponseType.CUSTOM_PLOTTING_FUNCTION:
            if not any(re.search(r'\b' + re.escape(func) + r'\b', value) for func in CUSTOM_PLOTTING_FUNCTIONS):
                raise ValueError("Invalid custom plotting function content")
        return value
    
    @root_validator(pre=True, allow_reuse=True)
    def validate_parameters(cls, values):
        response_type = values.get("response_type")
        response_content = values.get("response_content")
        
        def get_unique_values(df, column_name):
            return df[column_name].unique().tolist()

        if response_type == ResponseType.CUSTOM_PLOTTING_FUNCTION:
            df = values.get("df")  # Assume df is passed as a part of the values for validation
            unique_endpoints = get_unique_values(df, 'digital_EP')
            unique_visits = get_unique_values(df, 'VISIT')
            allowed_values = set(unique_endpoints + unique_visits + [True, False])

            # Extract parameters
            params = re.findall(r'\b\w+\b', response_content)
            for param in params:
                if param not in allowed_values:
                    param = param.upper()
        return values

def call_langchain_agent(df, model, session_state_messages, num_clarifying_questions_asked, max_clarifying_questions):
    """ 
    Function to handle clarifying questions and PandasAI prompts.
    """
    messages = get_system_message(df)
    summarized_messages = get_summarized_session_state_messages(session_state_messages)
    messages.extend(summarized_messages)
    
    if num_clarifying_questions_asked >= max_clarifying_questions:
        parser = JsonOutputParser(pydantic_object=Max_Questions_Exceeded_Response)
        messages.append({"role": "system", "content": "DO NOT ask any more clarifying questions. Proceed with the response."})
    else:
        parser = JsonOutputParser(pydantic_object=Response)
        
    # debugging
    st.write("The system_message is:")
    st.write(messages)
    
    prompt_template = PromptTemplate(
        template="{format_instructions}\n{messages}",
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
    
    if response["response_type"] == ResponseType.CLARIFICATION_QUESTION:
        num_clarifying_questions_asked += 1
        return response["response_content"], num_clarifying_questions_asked
    elif response["response_type"] == ResponseType.PANDASAI_PROMPT:
        # Add skills to the Agent
        vectorstore = ChromaDB()
        description = "You're a data analyst. IF you need to plot anything, you should provide the code to visualize the answer using plotly. Otherwise, provide a simple answer, no plot."
        agent = Agent(df, config={"llm": model, "response_parser": StreamlitResponse}, vectorstore=vectorstore, description = description)
        st.write(agent.last_code_executed)
        return agent.chat(response["response_content"]), num_clarifying_questions_asked
    elif response["response_type"] == ResponseType.CUSTOM_PLOTTING_FUNCTION:
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
        return result, num_clarifying_questions_asked