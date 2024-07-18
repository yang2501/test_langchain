# # Step 1: Import necessary modules and define Langchain tools
# from langchain.agents import Tool, tool
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.pydantic_v1 import BaseModel, Field
# import pandas as pd
# from pandasai import Agent

# # Azure OpenAI endpoint setup and authentication
# from setup_azure_openai import AzureOpenAISetup
# from langchain_openai.chat_models import AzureChatOpenAI

# azure_setup = AzureOpenAISetup()
# azure_setup.refresh_token()
# langchain_llm = AzureChatOpenAI(
#     streaming=False,
#     deployment_name="gpt-4-32k",
#     azure_endpoint="https://do-openai-instance.openai.azure.com/",
#     temperature=0
# )


# # Step 2: Create prompt
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are a very powerful assistant, you always ask context specific clarifying questions."),
#         ("user", "{input}"),
#         MessagesPlaceholder(variable_name="agent_scratchpad"),
#     ]
# )

# # Step 3: Define PandasAIPrompt and StandardizedDf classes
# class PandasAIPrompt(BaseModel):
#     prompt: str = Field()
    
# all_dfs_info_list: {pd.DataFrame} = {}
# current_dfs = []
# # Create a graph to represent data frame relationships using how they were transformed to create appropriate names that help with identifying the data frames the user needs/ is referencing for a specific query

# # Step 5: Define call_pandasai function
# @tool
# def call_pandasai(prompt: str):
#     """
#     Calls the pandasai agent with the clarified, prompt engineered, user prompt. The user prompt can now be parse for function parameters for these functions:
#     1. def bland_altman_plot(df, endpoint1, endpoint2, device1=None, device2=None, bySeverityCategory=False):
#     2. def change_from_baseline_plot(df, endpoint, device=None):
#     3. def plot_endpoint_distribution(df, endpoint, device1=None, device2=None, visit=None, bySeverityCategory=False):
#     4. def plot_correlation(df, endpoint1, endpoint2, device1=None, device2=None, bySeverityCategory=False):
#     5. def severity_category_confusion_matrix(df, endpoint, device=None, visit1='Screening', visit2=None):
#     6. def categorized_strip_plot(df, endpoint, gold_standard_device, visit=None):
    
#     If parameters are missing, ask clarifying questions.
#     """
    
#     # Initialize PandasAI agent if not already initialized
#     global current_dfs
#     if not current_dfs:
#         return "No data frame loaded. Please load a data frame first using pull_standardized_dataframe."
    
#     from setup_create_pandasai_agent import create_agent
#     df = pd.read_csv("./data/standardized_analysis_ready_df.csv")
#     df.drop(columns=['Unnamed: 0'], inplace=True)
#     pandasaiAgent = create_agent(df)
#     pandasaiAgent.chat(prompt)
#     return pandasaiAgent.last_code_executed

# @tool
# def pull_standardized_dataframe():
#     """
#     Dummy tool to pull standardized anlaysis ready data frame for pandasai to perform custom plotting functions on
#     """
#     # Load DataFrame 
#     df = pd.read_csv("./data/standardized_analysis_ready_df.csv")
#     df.drop(columns=['Unnamed: 0'], inplace=True)
#     current_dfs.append(df)
#     return "Standardized data frame loaded successfully."
 
# # # Tool to pull data frames for normal pandasai purposes using eu.query
# # @tool 
# # def retrieve_dataframe(prompt: str):
#     """
#     If the user wants to perform custom plotting functions, retrieve the standardized analysis ready dataframe using eu.query(). make sure to clarify the study name with the user
#     """
# #     llm.call_as_llm("ask a clarifying question to the user for which index they want if it's raw sensor data, call the dynamic aggregation function to create a special data frame before displaying to user.")

# # # Tool to pull standardized analysis ready data frames indices

# # @tool
# # def select_relevant_data_frames(prompt: str):
# #     llm.call_as_llm("Select the relevant data frames given the list of available data frames, the use char history and the user prompt.")

# # Tool for big data aggregation

# # Tool for anomaly detection

# # Tool for pattern recognition

# tools = [call_pandasai, pull_standardized_dataframe]

# # Step 8: Bind tools to the LLM
# llm = langchain_llm
# llm_with_tools = llm.bind_tools(tools)

# # Step 9: Add memory and set up chat history tracking
# MEMORY_KEY = "chat_history"
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "Your job is to get information from a user about what type of task they want to execute and execute it through function calls."),
#         MessagesPlaceholder(variable_name=MEMORY_KEY),
#         ("user", "{input}"),
#         MessagesPlaceholder(variable_name="agent_scratchpad"),
#     ]
# )

# # Track chat history
# from langchain_core.messages import AIMessage, HumanMessage
# chat_history = []

# # Step 10: Create the agent
# from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
# from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser

# agent = (
#     {
#         "input": lambda x: x["input"],
#         "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
#         "chat_history": lambda x: x["chat_history"],
#     }
#     | prompt
#     | llm_with_tools
#     | OpenAIToolsAgentOutputParser()
# )

# from langchain.agents import AgentExecutor
# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# # # Step 11: Invoke the agent for the desired task
# input1 = "Plot a bland altman plot comparing AHI for WatchPat and PSG. Don't plot by severity category."
# result = agent_executor.invoke({"input": input1, "chat_history": chat_history})

# chat_history.extend(
#     [
#         HumanMessage(content=input1),
#         AIMessage(content=result["output"]),
#     ]
# )
