from plotly.graph_objs._figure import Figure

def get_system_message(df):
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
        f"Use the uploaded dataframe. Don't ask questions regarding the dataframe."
        f"You are a helpful assistant specialized in identifying the user's intent and generating clarifying questions."
        f"If no clarifying questions are needed, output the custom function header OR Pandas.AI natural language prompt directly."
        f"ASK AS FEW CLARIFYING QUESTIONS AS POSSIBLE."
        f"When a user's prompt is too vague, create a context-specific clarifying question using the parameters in the function list. "
        f"For example, if the prompt likely requires calling a custom data visualization function, ask the user to specify the necessary parameters. "
        f"\nBelow is the list of custom plotting functions and their parameters: {function_headers}. "
        f"Below is more information about the data in the dataframe. This is useful for matching the prompt to a function and its parameters"
        f"The unique endpoints in the data frame are: {unique_endpoints}. "
        f"The unique devices in the data frame are: {unique_devices}. The unique visits in the data frame are: {unique_visits}. "
        f"All function header parameters MUST be a unique value in the data frame or the default value."
    )

    messages = [
        {"role": "system", "content": system_content},
    ]
    
    return messages
    
def get_summarized_session_state_messages(session_state_messages):
    def describe_plot(fig):
        # Create a one-sentence description for the plot
        return fig.layout.title.text if fig.layout.title.text else "Plotly figure"

    def copy_and_describe_messages(session_state_messages):
        described_messages = []
        for session_state_message in session_state_messages:
            message_copy = session_state_message.copy()
            if message_copy["role"] == "assistant" and all(isinstance(fig, Figure) for fig in message_copy["content"]):
                message_copy["content"] = [describe_plot(fig) for fig in session_state_message["content"]]
            described_messages.append(message_copy)
        return described_messages[-10:]
    
    return copy_and_describe_messages(session_state_messages)