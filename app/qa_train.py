# Will have to recheck/ redo the skills to make sure they perform accordingly to different datasets. 




# Instructions Training
# Correctly format the training document
training_docs = [
    "For each pandasai skills function where any parameter is missing. Ask the user a clarifying question whether they want to use the default plot or specify a parameter. For example, if the use didn't specify to set the bySeverityCategory to True/ False, ask the user: Would you like to plot by severity category?", 
    "Never return an input error, always ask the user to clarify function call parameters if mandatory ones are missing"
]


## Q/A train
# Train for bland_altman_plot
query = "Plot a bland altman plot comparing AHI for the devices"
query2 = "Generate the Bland-Altman plot for ahi using watchpat and psg devices"
query3 = "Generate the Bland-Altman plot for AHI for all severity categories"
response = """
def analyze_data(dfs: list[pd.DataFrame]) -> dict:
    df = dfs[0]
    devices = df['DEVICE'].unique()

    bland_altman_plot(df, endpoint1='AHI', endpoint2='AHI', device1=device[0], device2=device[1], bySeverityCategory=False)
    return { "type": "plot", "value": "temp_chart.png"}
"""
response2 = """
def analyze_data(dfs: list[pd.DataFrame]) -> dict:
    df = dfs[0]

    bland_altman_plot(df, endpoint1='AHI', endpoint2='AHI', device1='WatchPAT', device2='PSG', bySeverityCategory=False)
    return { "type": "plot", "value": "temp_chart.png"}
"""
response3 = """
def analyze_data(dfs: list[pd.DataFrame]) -> dict:
    df = dfs[0]
    devices = df['DEVICE'].unique()

    bland_altman_plot(df, endpoint1='AHI', endpoint2='AHI', device1=device[0], device2=device[1], bySeverityCategory=True)
    return { "type": "plot", "value": "temp_chart.png"}
"""

#Train for change from baseline
query4 = "Plot the change from baseline for AHI"
query5 = "Plot the AHI change from baseline for all devices"
response4 = """
def analyze_data(dfs: list[pd.DataFrame]) -> dict:
    df = dfs[0]
    
    change_from_baseline_plot(df, endpoint='AHI', device=None)
    return { "type": "plot", "value": "temp_chart.png"}
"""
response5 = """
def analyze_data(dfs: list[pd.DataFrame]) -> dict:
    df = dfs[0]
    
    change_from_baseline_plot(df, endpoint='AHI', device=None)
    return { "type": "plot", "value": "temp_chart.png"}
"""

query6 = "Plot the PSG AHI change from baseline"
response6 = """
def analyze_data(dfs: list[pd.DataFrame]) -> dict:
    df = dfs[0]
    
    change_from_baseline_plot(df, endpoint='AHI', device='PSG')
    return { "type": "plot", "value": "temp_chart.png"}
"""

# Train for plotting endpoint distribution
query7 = "Plot endpoint distribution of AHI for each device for each visit to have a general idea of device agreement"
query8 = "Plot AHI distribution by severity category"
response7 = """
def analyze_data(dfs: list[pd.DataFrame]) -> dict:
    df = dfs[0]
    devices = df['DEVICE'].unique()
    
    plot_endpoint_distribution(df, 'AHI', device1=devices[0], device2=devices[1], visit=None, bySeverityCategory = False)
    return { "type": "plot", "value": "temp_chart.png"}
"""
response8 = """
def analyze_data(dfs: list[pd.DataFrame]) -> dict:
    df = dfs[0]
    
    plot_endpoint_distribution(df, 'AHI', device1=None, device2=None, visit=None, bySeverityCategory = True)
    return { "type": "plot", "value": "temp_chart.png"}
"""

# Train for ploting correlation between two endpoints or devices
query9 = "Plot the correlation between the two devices for AHI"
query10 = "Plot AHI correlation comparing the devices"
response9 = """
def analyze_data(dfs: list[pd.DataFrame]) -> dict:
    df = dfs[0]
    
    devices = df['DEVICE'].unique()
    
    plot_correlation(df, endpoint1='AHI', endpoint2='AHI', device1=devices[0], device2=devices[1], bySeverityCategory=False)
    return { "type": "plot", "value": "temp_chart.png"}
"""
response10 = """
def analyze_data(dfs: list[pd.DataFrame]) -> dict:
    df = dfs[0]
    
    devices = df['DEVICE'].unique()
    
    plot_correlation(df, endpoint1='AHI', endpoint2='AHI', device1=devices[0], device2=devices[1], bySeverityCategory=False)
    return { "type": "plot", "value": "temp_chart.png"}
"""

# Train for plotting severity category confusion matrix
query11 = "Plot a confusion matrix visualizing how the severity category changes over time for AHI"
query12 = "Plot AHI severity confusion matrix"
response11 = """
def analyze_data(dfs: list[pd.DataFrame]) -> dict:
    df = dfs[0]
    
    severity_category_confusion_matrix(df, endpoint = 'AHI', device=None, visit1='Screening', visit2=None)
    return { "type": "plot", "value": "temp_chart.png"}
"""
response12 = """
def analyze_data(dfs: list[pd.DataFrame]) -> dict:
    df = dfs[0]
    
    severity_category_confusion_matrix(df, endpoint = 'AHI', device=None, visit1='Screening', visit2=None)
    return { "type": "plot", "value": "temp_chart.png"}
"""

# Train for creating a categorized strip plot
query13 = "Create a categorized strip plot for AHI where PSG is the gold standard"
response13 = """
def analyze_data(dfs: list[pd.DataFrame]) -> dict:
    df = dfs[0]
    
    categorized_strip_plot(df, endpoint='AHI', gold_standard_device='PSG', visit=None)
    return { "type": "plot", "value": "temp_chart.png"}
"""

# The model will use the information provided in the training to generate a response
from pandasai import Agent

def train_agent(agent: Agent):
    if agent is None:
        raise ValueError("Agent is not initialized")
    
    agent.train(docs=training_docs)
    agent.train(queries=[query, query2, query3, query4, query5, query6, query7, query8, query9, query10, query11, query12, query13], 
                codes=[response, response2, response3, response4, response5, response6, response7, response8, response9, response10, response11, response12, response13])
    return agent





