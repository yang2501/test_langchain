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
query2 = "Generate the Bland-Altman plot for comparing AHI and PAHI"
query3 = "Generate the Bland-Altman plot comparing AHI and PAHI for all severity categories"
response = """
def analyze_data(dfs: list[pd.DataFrame]) -> dict:
    import pandas as pd
    import plotly.io as pio

    df = dfs[0]
    fig = bland_altman_plot(df, endpoint1='AHI', endpoint2='PAHI', bySeverityCategory=False)
    # Convert Plotly figure to JSON
    fig_json = pio.to_json(fig)
    return fig_json
"""
response2 = """
def analyze_data(dfs: list[pd.DataFrame]) -> dict:
    import pandas as pd
    import plotly.io as pio

    df = dfs[0]
    fig = bland_altman_plot(df, endpoint1='AHI', endpoint2='PAHI', bySeverityCategory=False)
    # Convert Plotly figure to JSON
    fig_json = pio.to_json(fig)
    return fig_json
"""
response3 = """
def analyze_data(dfs: list[pd.DataFrame]) -> dict:
    import pandas as pd
    import plotly.io as pio

    df = dfs[0]
    fig = bland_altman_plot(df, endpoint1='AHI', endpoint2='PAHI', bySeverityCategory=True)
    # Convert Plotly figure to JSON
    fig_json = pio.to_json(fig)
    return fig_json
"""

#Train for change from baseline
query4 = "Plot the change from baseline for AHI"
query5 = "Plot the PAHI change from baseline for all devices"
response4 = """
    import pandas as pd

    df = dfs[0]
    change_from_baseline_plot(df, endpoint='AHI')
"""
response5 = """
    import pandas as pd

    df = dfs[0]
    change_from_baseline_plot(df, endpoint='PAHI')
"""

query6 = "Plot the PSG AHI change from baseline"
response6 = """
    import pandas as pd

    df = dfs[0]
    change_from_baseline_plot(df, endpoint='AHI')
"""

# # Train for plotting endpoint distribution
# query7 = "Plot endpoint distribution of AHI vs PAHI to have a general idea of device agreement"
# query8 = "Plot AHI vs PAHI distribution by severity category"
# response7 = """
# def analyze_data(dfs: list[pd.DataFrame]) -> dict:
#     df = dfs[0]
#     devices = df['DEVICE'].unique()
    
#     plot_endpoint_distribution(df, 'AHI', 'PAHI', bySeverityCategory = False)
#     return { "type": "plot", "value": "temp_chart.png"}
# """
# response8 = """
# def analyze_data(dfs: list[pd.DataFrame]) -> dict:
#     df = dfs[0]
    
#     plot_endpoint_distribution(df, 'AHI', 'PAHI', bySeverityCategory = True)
#     return { "type": "plot", "value": "temp_chart.png"}
# """

# # Train for ploting correlation between two endpoints or devices
# query9 = "Plot the correlation between AHI and PAHI"
# query10 = "Plot AHI vs PAHI correlation comparing the devices by severity category"
# response9 = """
# def analyze_data(dfs: list[pd.DataFrame]) -> dict:
#     df = dfs[0]
    
#     devices = df['DEVICE'].unique()
    
#     plot_correlation(df, endpoint1='AHI', endpoint2='PAHI', bySeverityCategory=False)
#     return { "type": "plot", "value": "temp_chart.png"}
# """
# response10 = """
# def analyze_data(dfs: list[pd.DataFrame]) -> dict:
#     df = dfs[0]
    
#     devices = df['DEVICE'].unique()
    
#     plot_correlation(df, endpoint1='AHI', endpoint2='PAHI', bySeverityCategory=True)
#     return { "type": "plot", "value": "temp_chart.png"}
# """

# # Train for plotting severity category confusion matrix
# query11 = "Plot a confusion matrix visualizing how the severity category changes over time for AHI"
# query12 = "Plot AHI severity confusion matrix"
# response11 = """
# def analyze_data(dfs: list[pd.DataFrame]) -> dict:
#     df = dfs[0]
    
#     severity_category_confusion_matrix(df, endpoint = 'AHI', visit1='Screening', visit2=None)
#     return { "type": "plot", "value": "temp_chart.png"}
# """
# response12 = """
# def analyze_data(dfs: list[pd.DataFrame]) -> dict:
#     df = dfs[0]
    
#     severity_category_confusion_matrix(df, endpoint = 'AHI', visit1='Screening', visit2=None)
#     return { "type": "plot", "value": "temp_chart.png"}
# """

# # Train for creating a categorized strip plot
# query13 = "Create a categorized strip plot for PAHI where AHI is the gold standard"
# response13 = """
# def analyze_data(dfs: list[pd.DataFrame]) -> dict:
#     df = dfs[0]
    
#     categorized_strip_plot(df, endpoint='PAHI', gold_standard_endpoint='AHI', visit=None)
#     return { "type": "plot", "value": "temp_chart.png"}
# """

# query14 = "Plot the correlation between AHI and PAHI. Include the equation of the line of best fit, the Pearson's correlation coefficient and the R squared value."
# response14 = """ 
# def analyze_data(dfs: list[pd.DataFrame]) -> dict:
#     df = dfs[0]
    
#     devices = df['DEVICE'].unique()
    
#     plot_correlation(df, endpoint1='AHI', endpoint2='PAHI', bySeverityCategory=False)
#     return { "type": "plot", "value": "temp_chart.png"}
# """

# query15 = "Create a visualization report comparing the endpoints AHI and PAHI where AHI is the gold standard."
# response15 = """ 
# def analyze_data(dfs: list[pd.DataFrame]) -> dict:
#     df = dfs[0]
    
#     devices = df['DEVICE'].unique()
    
#     two_endpoints_visualization_report(df, 'AHI', 'PAHI', gold_standard_endpoint='AHI', bySeverityCategory=False)
#     return { "type": "plot", "value": "temp_chart.png"}
# """

# query16 = "Create a visualization report comparing the endpoints AHI and PAHI where PSG is the gold standard device."
# response16 = """ 
# def analyze_data(dfs: list[pd.DataFrame]) -> dict:
#     df = dfs[0]
    
#     devices = df['DEVICE'].unique()
    
#     two_endpoints_visualization_report(df, 'AHI', 'PAHI', gold_standard_endpoint='AHI', bySeverityCategory=False)
#     return { "type": "plot", "value": "temp_chart.png"}
# """

def get_training_materials():
    
    docs=training_docs
    queries=[query, query2, query3, query4, query5, query6] #, query7, query8, query9, query10, query11, query12, query13, query14]
    codes=[response, response2, response3, response4, response5, response6] #, response7, response8, response9, response10, response11, response12, response13, response14]
    return docs, queries, codes





