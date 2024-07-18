# import streamlit as st
# import pandas as pd
# # Azure OpenAI endpoint setup and authentication
# from setup_azure_openai import AzureOpenAISetup
# from langchain_openai.chat_models import AzureChatOpenAI
# from pandasai import SmartDataframe, Agent
# from pandasai.ee.vectorstores import ChromaDB

# import os
# os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# azure_setup = AzureOpenAISetup()
# azure_setup.refresh_token()
# langchain_llm = AzureChatOpenAI(
#     streaming=False,
#     deployment_name="gpt-4-32k",
#     azure_endpoint="https://do-openai-instance.openai.azure.com/",
#     temperature=0
# )

# st.title("Original Working Data Analysis with PandasAI")
# uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

# from delete_cache import delete_cache_folder
# delete_cache_folder()

# if uploaded_file is not None:
#     data = pd.read_csv(uploaded_file)
#     st.write(data.head(3))
#     vectorstore=ChromaDB()
#     agent = Agent(data, config={"llm": langchain_llm}, vectorstore=vectorstore)
    
#     # Add skills to the SmartDataframe
#     from add_skills import bland_altman_plot, change_from_baseline_plot, plot_endpoint_distribution, plot_correlation, severity_category_confusion_matrix, categorized_strip_plot
#     agent.add_skills(bland_altman_plot, change_from_baseline_plot, plot_endpoint_distribution, plot_correlation, severity_category_confusion_matrix, categorized_strip_plot)
                    
#     # # QA train
#     from qa_train import get_training_materials
#     docs, queries, codes = get_training_materials()
#     agent.train(docs=docs)
#     agent.train(queries=queries, codes=codes)
                
#     prompt = st.text_area("Enter your prompt:")
    
#     if st.button("Generate"):
#         if prompt:
#             with st.spinner("Generating response..."):
#                 st.write(agent.chat(prompt))
#         else:
#             st.warning("Please enter a prompt")
    
