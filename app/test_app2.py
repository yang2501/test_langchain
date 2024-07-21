import streamlit as st

# Azure OpenAI endpoint setup and authentication
from setup_azure_openai import AzureOpenAISetup
from langchain_openai.chat_models import AzureChatOpenAI

azure_setup = AzureOpenAISetup()
azure_setup.refresh_token()
langchain_llm = AzureChatOpenAI(
    streaming=False,
    deployment_name="gpt-4-32k",
    azure_endpoint="https://do-openai-instance.openai.azure.com/",
    temperature=0
)

st.title("ChatGPT-like clone")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = langchain_llm.agenerate(prompt)
        st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})