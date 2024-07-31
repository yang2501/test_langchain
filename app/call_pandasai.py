# Import things that are needed generically
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool
import pandas as pd
from enum import Enum
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import streamlit as st


class PandasAIUserPrompt(BaseModel):
    query: str = Field(description="should be a prompt for pandasai")
 

def call_pandasai(df, model, session_state_messages, num_clarifying_questions_asked, max_clarifying_questions) -> str:
    """Calls pandasai for anomaly detection analysis and basic data analysis."""
    # Pandasai imports
    from pandasai import Agent
    from pandasai.ee.vectorstores import ChromaDB
    from pandasai.responses.streamlit_response import StreamlitResponse
    
    class CustomPandasAIResponseType(str, Enum):
       # CLARIFICATION_QUESTION = "clarification_question"
        PANDASAI_PROMPT = "pandasai_prompt"
 
    class CustomPandasAIResponse(BaseModel):
        response_type: CustomPandasAIResponseType = Field(description="can hold enum values: pandasai_prompt")
        response_content: str = Field(description="the response content being: the actual clarification question OR prompt to pandasai with FULL CONTEXT")
 
    parser = JsonOutputParser(pydantic_object=CustomPandasAIResponse)
    
    
    system_content = (
        f"Use the current dataframe."
        f"You are a data analyst specialized in analyzing anomalies and basic data analysis and visualizations."
       # f"If the user's question is too vague, ask a clarifying question."
      #  f"If no clarifying questions are needed, output the pandasai prompt."
       #    f"ASK AS FEW CLARIFYING QUESTIONS AS POSSIBLE."
    )
 
    messages = [
        {"role": "system", "content": system_content},
    ]
    
    messages.extend(session_state_messages)
 
    prompt_template = PromptTemplate(
        template="{format_instructions}\n{messages}",
        input_variables=["messages"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    chain = prompt_template | model | parser
    response = chain.invoke({"messages": messages})
    
    #if response["response_type"] == CustomPandasAIResponseType.CLARIFICATION_QUESTION:
       # num_clarifying_questions_asked += 1
       # return response["response_content"], num_clarifying_questions_asked
    if response["response_type"] == CustomPandasAIResponseType.PANDASAI_PROMPT:
        from anom_detect_skills import detect_anomalies, getDayDf, getNightDf, numberDayDf, numberNightDf, strtTimeDf, dayVsNightDf, freqTimeDf, avgScoreDf, plotDf, numAnomScoreDf
        from anom_detect_qa_train import get_training_materials
 
        vectorstore = ChromaDB()
        description = "You're a data analyst. IF you need to plot anything, you should provide the code to visualize the answer using plotly. Otherwise, provide a simple answer, no plot."
        agent = Agent(df, config={"llm": model, "response_parser": StreamlitResponse}, vectorstore=vectorstore, description = description)
        agent.add_skills(detect_anomalies, getDayDf, getNightDf, numberDayDf, numberNightDf, strtTimeDf, dayVsNightDf, freqTimeDf, avgScoreDf, plotDf, numAnomScoreDf)
        queries, codes = get_training_materials()
        
        agent.train(queries=queries, codes=codes)

        return agent.chat(response["response_content"]), agent.last_code_generated
 