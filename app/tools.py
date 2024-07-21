# Import things that are needed generically
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from langchain.agents.agent_types import AgentType

from setup_azure_openai import AzureOpenAISetup
from langchain_openai.chat_models import AzureChatOpenAI

import pandas as pd


azure_setup = AzureOpenAISetup()
azure_setup.refresh_token()
langchain_llm = AzureChatOpenAI(
    streaming=False,
    deployment_name="gpt-4-32k",
    azure_endpoint="https://do-openai-instance.openai.azure.com/",
    temperature=0
)

class PlotInput1(BaseModel):
    df: pd.DataFrame = Field(description= "Pandas dataframe containing the columns: 'VISIT': Visit name (e.g. VISIT2), 'USUBJID': Unique subject ID, 'digital_EP': Endpoint name (e.g. WASO, AHI, etc.), 'digital_EP_value': Endpoint value (some numeric value), 'digital_EP_severity_category': Severity category of the endpoint, 'COHORT': Treatment group (e.g. Placebo/ Treatment), 'DEVICE': Device name (e.g. WatchPAT, PSG, etc.)")
    endpoint1: str = Field(description="The first endpoint to compare (e.g., WASO)")
    endpoint2: str = Field(description="The second endpoint to compare (e.g., AHI)")
    bySeverityCategory: bool = Field(description="Optional parameter, Default is False, Whether to plot the plots by severity category.")
    
    class Config:
        arbitrary_types_allowed = True

class PlotInput2(BaseModel):
    df: pd.DataFrame = Field(description= "Pandas dataframe containing the columns: 'VISIT': Visit name (e.g. VISIT2), 'USUBJID': Unique subject ID, 'digital_EP': Endpoint name (e.g. WASO, AHI, etc.), 'digital_EP_value': Endpoint value (some numeric value), 'digital_EP_severity_category': Severity category of the endpoint, 'COHORT': Treatment group (e.g. Placebo/ Treatment), 'DEVICE': Device name (e.g. WatchPAT, PSG, etc.)")
    endpoint: str = Field(description="The endpoint to plot (e.g., WASO)")
    
    class Config:
        arbitrary_types_allowed = True

# Plot bland_altman to compare endpoint alignment

@tool
def bland_altman_plot(df, endpoint1, endpoint2, bySeverityCategory=False) -> str:
    """
    Generates a Bland-Altman plot to compare two endpoints, optionally by severity category.

    Parameters
    ----------
    df : DataFrame
        The original DataFrame containing the data.
            'VISIT': Visit name (e.g. VISIT2)
            'USUBJID': Unique subject ID
            'digital_EP': Endpoint name (e.g. WASO, AHI, etc.)
            'digital_EP_value': Endpoint value (some numeric value)
            'digital_EP_severity_category': Severity category of the endpoint
            'COHORT': Treatment group (e.g. Placebo/ Treatment)
            'DEVICE': Device name (e.g. WatchPAT, PSG, etc.)
    endpoint1 : str
        The first endpoint to compare (e.g., WASO).
    endpoint2 : str
        The second endpoint to compare (e.g., AHI).
    bySeverityCategory : bool, optional
        Whether to plot the Bland-Altman plots by severity category.

    Returns
    ----------
    JSON representation of the plotly figures.
    """
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.io as pio

    def plot_bland_altman(df, category=None):
        # Calculate the mean difference and limits of agreement
        mean_diff = df['difference'].mean()
        std_diff = df['difference'].std()
        upper_limit = mean_diff + 1.96 * std_diff
        lower_limit = mean_diff - 1.96 * std_diff
        count = len(df['difference'])

        # Create the Bland-Altman plot using Plotly
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df['mean'],
            y=df['difference'],
            mode='markers',
            marker=dict(color='blue', opacity=0.5),
            name='Data points'
        ))

        fig.add_trace(go.Scatter(
            x=df['mean'],
            y=[mean_diff] * len(df),
            mode='lines',
            line=dict(color='gray', dash='solid'),
            name=f'Mean Difference: {mean_diff:.2f}'
        ))

        fig.add_trace(go.Scatter(
            x=df['mean'],
            y=[upper_limit] * len(df),
            mode='lines',
            line=dict(color='red', dash='dash'),
            name=f'+1.96 SD: {upper_limit:.2f}'
        ))

        fig.add_trace(go.Scatter(
            x=df['mean'],
            y=[lower_limit] * len(df),
            mode='lines',
            line=dict(color='red', dash='dash'),
            name=f'-1.96 SD: {lower_limit:.2f}'
        ))

        title = f'Bland-Altman Plot for {count} {endpoint1} - {endpoint2} pairs'
        if category:
            title += f'\nSeverity Category: {category}'

        fig.update_layout(
            title=title,
            xaxis_title='Means',
            yaxis_title=f'Difference: {endpoint1} - {endpoint2}',
            showlegend=True
        )
        
        # Convert Plotly figure to JSON
        fig_json = pio.to_json(fig)
        return fig_json



    # Filter the DataFrame for the specified endpoints
    df1 = df[df['digital_EP'] == endpoint1][['USUBJID', 'VISIT', 'digital_EP_value', 'digital_EP_severity_category']].rename(columns={'digital_EP_value': f'{endpoint1}_value'})
    df2 = df[df['digital_EP'] == endpoint2][['USUBJID', 'VISIT', 'digital_EP_value', 'digital_EP_severity_category']].rename(columns={'digital_EP_value': f'{endpoint2}_value'})

    # Merge the two DataFrames on the subject ID and visit
    df_merged = pd.merge(df1, df2, on=['USUBJID', 'VISIT', 'digital_EP_severity_category'])

    # Check if there is data to plot
    if df_merged.empty:
        print("No overlapping data between the specified endpoints/devices.")
        return False, "No overlapping data to plot."

    plot_results = {}

    if bySeverityCategory:
        # Get the unique severity categories
        severity_categories = df_merged['digital_EP_severity_category'].unique()

        # Create a plot for each severity category
        for category in severity_categories:
            category_data = df_merged[df_merged['digital_EP_severity_category'] == category]

            # Calculate the mean and difference of the two endpoints
            category_data['mean'] = category_data[[f'{endpoint1}_value', f'{endpoint2}_value']].mean(axis=1)
            category_data['difference'] = category_data[f'{endpoint1}_value'] - category_data[f'{endpoint2}_value']

            success, fig_json = plot_bland_altman(category_data, category)
            plot_results[category] = fig_json

    else:
        # Calculate the mean and difference of the two endpoints
        df_merged['mean'] = df_merged[[f'{endpoint1}_value', f'{endpoint2}_value']].mean(axis=1)
        df_merged['difference'] = df_merged[f'{endpoint1}_value'] - df_merged[f'{endpoint2}_value']

        success, fig_json = plot_bland_altman(df_merged)
        plot_results["overall"] = fig_json

    return True, plot_results

plot_bland_altman = StructuredTool.from_function(
    func=bland_altman_plot,
    name="Plot bland altman",
    description="Plots a bland altman plot",
    args_schema=PlotInput1,
    return_direct=True,
    # coroutine= ... <- you can specify an async method if desired as well
)

# Plot change from baseline. By default, if no device is specified, it plots the change from baseline for an endpoint for all devices.

@tool
def change_from_baseline_plot(df, endpoint) -> str:
    """
    Plots a change from baseline chart for different cohorts.

    Parameters
    ----------
    df : DataFrame
        The original DataFrame containing the data.
            'VISIT' (e.g. VISIT3). The get_visit_number() function relies on screening visit being marked as 'Screening'. The data should be cleaned such that there are only valid visits in this column.
            'USUBJID': unique subject ID.
            'digital_EP': (e.g. WASO, AHI, etc.).
            'digital_EP_value': (some numeric value).
            'digital_EP_severity_category': severity category of the endpoint.
            'COHORT': treatment group (e.g. Placebo/ Treatment).
            'DEVICE': device name (e.g. WatchPAT, PSG, etc.).
    endpoint : str
        The Digital_EP to plot (e.g., WASO, AHI, etc.).

    Returns
    ----------
    None
    """
    import pandas as pd
    import re
    import plotly.graph_objects as go

    def get_visit_number(visit):
        """
        Extracts the numeric part of the visit name for sorting.
        Non-numeric visits are considered invalid and return None.
        """
        match = re.match(r'^VISIT(\d+)$', visit, re.IGNORECASE)
        if match:
            return int(match.group(1))
        elif visit.lower() == 'screening':
            return 0
        else:
            return None

    # Filter by endpoint
    df_filtered = df[df['digital_EP'] == endpoint]

    if df_filtered.empty:
        print("No data available for the specified endpoint.")
        return

    # Extract the numeric part of the VISIT string
    df_filtered['visit_num'] = df_filtered['VISIT'].apply(get_visit_number)
    df_filtered = df_filtered.dropna(subset=['visit_num'])

    # Ensure visit_num is integer type
    df_filtered['visit_num'] = df_filtered['visit_num'].astype(int)

    if df_filtered['visit_num'].isnull().all():
        print("No valid visits found.")
        return

    max_visit_row = df_filtered.loc[df_filtered['visit_num'].idxmax()]
    comparison_visit = max_visit_row['VISIT']

    # Filter the dataframe for the specified endpoint and remove rows with unknown cohort
    df_filtered = df_filtered[df_filtered['COHORT'] != 'Unknown']

    # Calculate the baseline value for each subject
    df_baseline = df_filtered[df_filtered['VISIT'].str.lower() == 'screening'][['USUBJID', 'digital_EP_value']]
    df_baseline = df_baseline.rename(columns={'digital_EP_value': 'digital_EP_baseline_value'})

    # Merge baseline values with the original dataframe
    df_merged = df_filtered.merge(df_baseline, on='USUBJID', how='left')

    # Calculate the change from baseline for each subject at each visit
    df_merged['calculated_change_from_baseline'] = df_merged['digital_EP_value'] - df_merged['digital_EP_baseline_value']

    # Group by cohort and visit to calculate the mean, sem, and number of subjects
    grouped = df_merged.groupby(['COHORT', 'VISIT'])
    aggDf = grouped.agg(
        mean=('calculated_change_from_baseline', 'mean'),
        sem=('calculated_change_from_baseline', lambda x: x.std() / (len(x) ** 0.5)),
        num_subjects=('calculated_change_from_baseline', 'count')
    ).reset_index()
    
    cohortList = aggDf['COHORT'].unique().tolist()

    cohort2Color = {
        cohortList[0]: 'black',
        cohortList[1]: 'red'
    }

    fig = go.Figure()

    xticks = []
    xlabels = []

    for cohort in cohortList:
        data = aggDf[aggDf['COHORT'] == cohort].reset_index(drop=True)

        if data.empty:
            continue

        data['visit_num'] = data['VISIT'].apply(get_visit_number)
        data = data.sort_values(by=['visit_num'])
        visits = data['VISIT'].tolist()

        x = data['visit_num'].tolist()
        xticks += x
        xlabels += visits

        y = data['mean'].tolist()
        yerr = data['sem'].tolist()

        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            error_y=dict(type='data', array=yerr, visible=True),
            mode='lines+markers',
            marker=dict(size=8),
            line=dict(color=cohort2Color[cohort]),
            name=cohort
        ))

    xticks = list(set(xticks))
    xticks.sort()

    unique_xlabels = []
    unique_xticks = []
    for tick, label in zip(xticks, xlabels):
        if tick not in unique_xticks:
            unique_xticks.append(tick)
            unique_xlabels.append(label)

    fig.update_layout(
        title=f"{df.loc[df['digital_EP'] == endpoint, 'DEVICE'].iloc[0] if not df.loc[df['digital_EP'] == endpoint, 'DEVICE'].empty else 'Unknown Device'} {endpoint} Change from Baseline",
        xaxis=dict(
            title='Visit',
            tickmode='array',
            tickvals=unique_xticks,
            ticktext=unique_xlabels,
            tickangle=45
        ),
        yaxis=dict(title=f'{endpoint} Change'),
        legend=dict(title='Cohort'),
        template='plotly_white'
    )

    fig.show()

plot_change_from_baseline = StructuredTool.from_function(
    func=change_from_baseline_plot,
    name="Plot change from baseline",
    description="Plots the change from baseline",
    args_schema=PlotInput2,
    return_direct=True,
    # coroutine= ... <- you can specify an async method if desired as well    
)

df = pd.read_csv("/Users/L075945/Library/CloudStorage/OneDrive-EliLillyandCompany/Desktop/langchain_app/data/standardized_analysis_ready_df-1.csv")

tools = [plot_bland_altman, plot_change_from_baselinelocals]

agent = create_pandas_dataframe_agent(
    ChatOpenAI(temperature=0, model=langchainllm),
    df,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    extra_tools = tools
)

# # Get the prompt to use - you can modify this!
# prompt = hub.pull("hwchase17/openai-functions-agent")

# # Construct the OpenAI Functions agent
# agent = create_openai_functions_agent(langchain_llm, tools, prompt)

# # Create an agent executor by passing in the agent and tools
# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent.invoke({"input": "Plot a bland altman plot comparing ahi and pahi by severity category. Using this dataframe: df"})

# # Plot endpoint distribution for each device for each visit to have a general idea of device agreement

# @skill
# def original_plot_endpoint_distribution(df, endpoint1, endpoint2, bySeverityCategory=False):
#     """
#     Plots histograms showing the distribution of specified endpoints for each device and compares the means.
#     Optionally, plots the distribution by severity category.

#     Parameters
#     ----------
#     df : DataFrame
#         The original DataFrame containing the data.
#             'VISIT': Visit name (e.g. VISIT2)
#             'USUBJID': Unique subject ID
#             'digital_EP': Endpoint name (e.g. WASO, AHI, etc.)
#             'digital_EP_value': Endpoint value (some numeric value)
#             'digital_EP_severity_category': Severity category of the endpoint
#             'COHORT': Treatment group (e.g. Placebo/ Treatment)
#             'DEVICE': Device name (e.g. WatchPAT, PSG, etc.)
#     endpoint1 : str
#         The first endpoint to plot (e.g., WASO, AHI, etc.).
#     endpoint2 : str
#         The second endpoint to plot (e.g., WASO, AHI, etc.).
#     bySeverityCategory : bool, optional
#         Whether to plot the distribution by severity category.

#     Returns
#     ----------
#     None
#     """
    
#     import pandas as pd
#     from scipy.stats import ttest_ind
#     import numpy as np
#     import plotly.figure_factory as ff
    
#     def get_visit_number(visit):
#         """
#         Extracts the numeric part of the visit name for sorting.
#         Non-numeric visits are considered invalid and return None.
#         """
#         import re
#         match = re.match(r'^VISIT(\d+)$', visit, re.IGNORECASE)
#         if match:
#             return int(match.group(1))
#         elif visit.lower() == 'screening':
#             return 0
#         else:
#             return None

#     # Filter the DataFrame for the specified endpoints
#     df_endpoint1 = df[df['digital_EP'] == endpoint1]
#     df_endpoint2 = df[df['digital_EP'] == endpoint2]

#     if bySeverityCategory:
#         # Get the unique severity categories
#         severity_categories = df['digital_EP_severity_category'].dropna().unique()

#         for category in severity_categories:
#             category_data1 = df_endpoint1[df_endpoint1['digital_EP_severity_category'] == category]
#             category_data2 = df_endpoint2[df_endpoint2['digital_EP_severity_category'] == category]
            
#             hist_data1 = [category_data1['digital_EP_value'].dropna().values]
#             hist_data2 = [category_data2['digital_EP_value'].dropna().values]
#             group_labels = [f'{category} {endpoint1}', f'{category} {endpoint2}']

#             fig1 = ff.create_distplot(hist_data1, [group_labels[0]], bin_size=0.2, show_hist=False, show_rug=False)
#             fig1.update_layout(title=f'Distribution of {endpoint1} by Severity Category: {category}', template='plotly_white')
#             fig1.show()

#             fig2 = ff.create_distplot(hist_data2, [group_labels[1]], bin_size=0.2, show_hist=False, show_rug=False)
#             fig2.update_layout(title=f'Distribution of {endpoint2} by Severity Category: {category}', template='plotly_white')
#             fig2.show()
#     else:
#         # Filter out invalid visits
#         df_endpoint1['visit_num'] = df_endpoint1['VISIT'].apply(get_visit_number)
#         df_endpoint2['visit_num'] = df_endpoint2['VISIT'].apply(get_visit_number)
#         df_endpoint1 = df_endpoint1.dropna(subset=['visit_num'])
#         df_endpoint2 = df_endpoint2.dropna(subset=['visit_num'])

#         # Ensure visit_num is integer type
#         df_endpoint1['visit_num'] = df_endpoint1['visit_num'].astype(int)
#         df_endpoint2['visit_num'] = df_endpoint2['visit_num'].astype(int)

#         # Identify visits where data for both endpoints is available
#         visits = sorted(set(df_endpoint1['VISIT'].unique()).intersection(df_endpoint2['VISIT'].unique()), key=lambda x: get_visit_number(x))
        
#         hist_data1 = []
#         hist_data2 = []
#         group_labels1 = []
#         group_labels2 = []

#         for visit in visits:
#             visit_data1 = df_endpoint1[df_endpoint1['VISIT'].str.lower() == visit.lower()]['digital_EP_value'].dropna().values
#             visit_data2 = df_endpoint2[df_endpoint2['VISIT'].str.lower() == visit.lower()]['digital_EP_value'].dropna().values

#             if len(visit_data1) > 0:
#                 hist_data1.append(visit_data1)
#                 group_labels1.append(f'{visit} {endpoint1}')
            
#             if len(visit_data2) > 0:
#                 hist_data2.append(visit_data2)
#                 group_labels2.append(f'{visit} {endpoint2}')

#         if hist_data1:
#             fig1 = ff.create_distplot(hist_data1, group_labels1, bin_size=0.2, show_hist=False, show_rug=False)
#             fig1.update_layout(title=f'Distribution of {endpoint1} values across visits', template='plotly_white')
#             fig1.show()

#         if hist_data2:
#             fig2 = ff.create_distplot(hist_data2, group_labels2, bin_size=0.2, show_hist=False, show_rug=False)
#             fig2.update_layout(title=f'Distribution of {endpoint2} values across visits', template='plotly_white')
#             fig2.show()

# # Plot endpoint correlation

# @skill
# def plot_correlation(df, endpoint1, endpoint2, bySeverityCategory=False):
#     """
#     Plots scatter plots showing the correlation between two endpoints for each device.
#     Optionally, plots the correlation by severity category.

#     Parameters
#     ----------
#     df : DataFrame
#         The original DataFrame containing the data.
#             'VISIT': Visit name (e.g. VISIT2)
#             'USUBJID': Unique subject ID
#             'digital_EP': Endpoint name (e.g. WASO, AHI, etc.)
#             'digital_EP_value': Endpoint value (some numeric value)
#             'digital_EP_severity_category': Severity category of the endpoint
#             'COHORT': Treatment group (e.g. Placebo/ Treatment)
#             'DEVICE': Device name (e.g. WatchPAT, PSG, etc.)
#     endpoint1 : str
#         The first endpoint to compare (e.g., WASO).
#     endpoint2 : str
#         The second endpoint to compare (e.g., AHI).
#     bySeverityCategory : bool, optional
#         Whether to plot the correlation by severity category.

#     Returns
#     ----------
#     None
#     """
#     import pandas as pd
#     from scipy.stats import linregress, spearmanr, pearsonr
#     import plotly.graph_objects as go
#     import numpy as np

#     # Filter the DataFrame for the specified endpoints
#     df_endpoint1 = df[df['digital_EP'] == endpoint1]
#     df_endpoint2 = df[df['digital_EP'] == endpoint2]

#     # Ensure uniqueness by aggregating values within each group
#     df_endpoint1 = df_endpoint1.groupby(['USUBJID', 'VISIT', 'digital_EP_severity_category']).agg(
#         digital_EP_value=(f'digital_EP_value', 'mean')).reset_index()
#     df_endpoint2 = df_endpoint2.groupby(['USUBJID', 'VISIT', 'digital_EP_severity_category']).agg(
#         digital_EP_value=(f'digital_EP_value', 'mean')).reset_index()

#     # Merge the data for both endpoints to include only subjects with data for both
#     common_subjects = pd.merge(
#         df_endpoint1,
#         df_endpoint2,
#         on=['USUBJID', 'VISIT', 'digital_EP_severity_category'],
#         suffixes=(f'_{endpoint1}', f'_{endpoint2}')
#     )

#     # Check if there is data to plot
#     if common_subjects.empty:
#         print("No overlapping data between the specified endpoints/devices.")
#         return "No overlapping data to plot."

#     def create_plot(x, y, title, xlabel, ylabel):
#         # Calculate the correlation coefficients and line of best fit
#         slope, intercept, r_value, p_value, std_err = linregress(x, y)
#         line = slope * np.array(x) + intercept
#         pearson_corr, pearson_p_value = pearsonr(x, y)

#         # Create the scatter plot
#         fig = go.Figure()

#         fig.add_trace(go.Scatter(
#             x=x,
#             y=y,
#             mode='markers',
#             marker=dict(color='blue', opacity=0.7),
#             name='Data Points'
#         ))

#         fig.add_trace(go.Scatter(
#             x=x,
#             y=line,
#             mode='lines',
#             line=dict(color='red'),
#             name=f'Line of Best Fit: y={slope:.2f}x+{intercept:.2f}'
#         ))

#         fig.update_layout(
#             title=f'Pearson Correlation: {pearson_corr:.2f}, p-value: {pearson_p_value:.4f}',
#             xaxis_title=xlabel,
#             yaxis_title=ylabel,
#             template='plotly_white'
#         )

#         fig.show()

#     if bySeverityCategory:
#         # Get the unique severity categories
#         severity_categories = common_subjects['digital_EP_severity_category'].unique()

#         # Create a plot for each severity category
#         for category in severity_categories:
#             category_data = common_subjects[common_subjects['digital_EP_severity_category'] == category]

#             x = category_data[f'digital_EP_value_{endpoint1}']
#             y = category_data[f'digital_EP_value_{endpoint2}']

#             create_plot(x, y,
#                         f'{endpoint1} vs {endpoint2} Correlation<br>Severity Category: {category}',
#                         f'{endpoint1} Value', f'{endpoint2} Value')
#     else:
#         x = common_subjects[f'digital_EP_value_{endpoint1}']
#         y = common_subjects[f'digital_EP_value_{endpoint2}']

#         create_plot(x, y,
#                     f'{endpoint1} vs {endpoint2} Correlation',
#                     f'{endpoint1} Value', f'{endpoint2} Value')

# # Plot confusion matrix by severity category

# @skill
# def severity_category_confusion_matrix(df, endpoint, visit1='Screening', visit2=None):
#     """
#     Generates confusion matrices for severity categories of an endpoint between two visits for each treatment cohort.

#     Parameters
#     ----------
#     df : DataFrame
#         The original DataFrame containing the data.
#             'VISIT': Visit name (e.g. VISIT2)
#             'USUBJID': Unique subject ID
#             'digital_EP': Endpoint name (e.g. WASO, AHI, etc.)
#             'digital_EP_value': Endpoint value (some numeric value)
#             'digital_EP_severity_category': Severity category of the endpoint
#             'COHORT': Treatment group (e.g. Placebo/ Treatment)
#             'DEVICE': Device name (e.g. WatchPAT, PSG, etc.)
#     endpoint : str
#         The endpoint to analyze (e.g., WASO, AHI, etc.).
#     visit1 : str, optional
#         The first visit to compare (default is 'Screening').
#     visit2 : str, optional
#         The second visit to compare (default is the latest valid visit).

#     Returns
#     ----------
#     None
#     """
    
#     import pandas as pd
#     import plotly.figure_factory as ff

#     # Determine the latest visit if visit2 is not specified
#     if visit2 is None:
#         visits = df['VISIT'].str.extract(r'(\d+)', expand=False).dropna().astype(int)
#         latest_visit_num = visits.max()
#         visit2 = f'VISIT{latest_visit_num}'

#     # Filter the DataFrame for the specified endpoint
#     df_filtered = df[df['digital_EP'] == endpoint]

#     # Get the unique treatment cohorts
#     cohorts = df_filtered['COHORT'].unique()

#     # Define the order of severity categories
#     severity_order = ['No', 'Mild', 'Moderate', 'Severe']

#     for cohort in cohorts:
#         df_cohort = df_filtered[df_filtered['COHORT'] == cohort]

#         # Filter data for visit1 and visit2
#         df_visit1 = df_cohort[df_cohort['VISIT'].str.lower() == visit1.lower()][['USUBJID', 'digital_EP_severity_category']]
#         df_visit2 = df_cohort[df_cohort['VISIT'].str.lower() == visit2.lower()][['USUBJID', 'digital_EP_severity_category']]

#         # Rename columns to avoid confusion when merging
#         df_visit1 = df_visit1.rename(columns={'digital_EP_severity_category': f'severity_category_{visit1}'})
#         df_visit2 = df_visit2.rename(columns={'digital_EP_severity_category': f'severity_category_{visit2}'})

#         # Merge the two visits on USUBJID
#         df_merged = pd.merge(df_visit1, df_visit2, on='USUBJID')

#         if df_merged.empty:
#             print(f"No overlapping data between {visit1} and {visit2} for cohort {cohort}.")
#             continue

#         # Generate the confusion matrix
#         y_true = pd.Categorical(df_merged[f'severity_category_{visit1}'], categories=severity_order, ordered=True)
#         y_pred = pd.Categorical(df_merged[f'severity_category_{visit2}'], categories=severity_order, ordered=True)

#         cm = pd.crosstab(y_true, y_pred, rownames=[f'Severity Category at {visit1}'], colnames=[f'Severity Category at {visit2}'], dropna=False)

#         if cm.empty:
#             print(f"No data to plot for cohort {cohort}.")
#             continue

#         # Calculate percentages
#         cm_percentage = cm.div(cm.sum(axis=1), axis=0) * 100

#         # Create annotations with counts and percentages
#         annot = cm.astype(str) + "\n" + cm_percentage.round(2).astype(str) + '%'

#         # Create the confusion matrix plot using Plotly
#         z = cm.values
#         x = severity_order
#         y = severity_order

#         fig = ff.create_annotated_heatmap(
#             z,
#             x=x,
#             y=y,
#             annotation_text=annot.values,
#             colorscale='Blues',
#             showscale=True,
#             hoverinfo="z"
#         )

#         fig.update_layout(
#             title={
#                 'text': f'Confusion Matrix of Severity Categories for {endpoint} for the {cohort} Cohort',
#                 'y': 0.98,  # Adjust this value to move the title higher or lower
#                 'x': 0.5,
#                 'xanchor': 'center',
#                 'yanchor': 'top'
#             },
#             xaxis=dict(title=f'Severity Category at {visit2}', side='top'),
#             yaxis=dict(title=f'Severity Category at {visit1}', autorange='reversed')  # Reverse Y axis to start from "No" at the origin
#         )

#         fig.show()

# # Categorized Strip Plot
# @skill
# def categorized_strip_plot(df, endpoint, gold_standard_endpoint, visit=None):
#     """
#     Creates categorized strip plots for each device to visualize incorrect classifications compared to a gold standard device.

#     Parameters
#     ----------
#     df : DataFrame
#         The original DataFrame containing the data.
#             'VISIT': Visit name (e.g. VISIT2)
#             'USUBJID': Unique subject ID
#             'digital_EP': Endpoint name (e.g. WASO, AHI, etc.)
#             'digital_EP_value': Endpoint value (some numeric value)
#             'digital_EP_severity_category': Severity category of the endpoint
#             'COHORT': Treatment group (e.g. Placebo/ Treatment)
#             'DEVICE': Device name (e.g. WatchPAT, PSG, etc.)
#     endpoint : str
#         The endpoint to analyze (e.g., WASO, AHI, etc.).
#     gold_standard_endpoint : str
#         The gold standard endpoint to compare against.
#     visit : str, optional
#         The visit to filter the data (default is to use all visits).

#     Returns
#     ----------
#     None
#     """
    
#     import pandas as pd
#     import plotly.graph_objects as go
    
#     # Define severity order
#     severity_order = ['No', 'Mild', 'Moderate', 'Severe']
#     severity_colors = {category: color for category, color in zip(severity_order, ['green', 'yellow', 'orange', 'red'])}
    
#     # Filter the DataFrame for the specified endpoint and visit
#     df_filtered = df[(df['digital_EP'] == endpoint) | (df['digital_EP'] == gold_standard_endpoint)]
    
#     if visit:
#         df_filtered = df_filtered[df_filtered['VISIT'].str.lower() == visit.lower()]

#     # Pivot the DataFrame to compare endpoints
#     df_pivot = df_filtered.pivot_table(index='USUBJID', columns='digital_EP', values='digital_EP_severity_category', aggfunc='first')

#     # Check if both endpoints are in the DataFrame
#     if endpoint not in df_pivot.columns or gold_standard_endpoint not in df_pivot.columns:
#         print(f"Both {endpoint} and {gold_standard_endpoint} must be present in the DataFrame.")
#         return
    
#     # Create a DataFrame for plotting
#     plot_df = df_pivot[[gold_standard_endpoint, endpoint]].dropna().reset_index()

#     fig = go.Figure()

#     # Add the strip plot for the specified endpoint, placing them in the gold standard severity category bucket
#     for true_severity_index, true_severity in enumerate(severity_order):
#         subset = plot_df[plot_df[gold_standard_endpoint] == true_severity]
        
#         # Sort the subset by the endpoint severity to group colors together
#         sorted_subset = subset.sort_values(by=endpoint, key=lambda col: col.map({sev: i for i, sev in enumerate(severity_order)}))
        
#         # Assign x-positions within the bucket
#         for predicted_severity_index, predicted_severity in enumerate(severity_order):
#             sev_subset = sorted_subset[sorted_subset[endpoint] == predicted_severity]
#             x_positions = [true_severity_index + (predicted_severity_index - 1.5) * 0.1 for _ in range(len(sev_subset))]

#             fig.add_trace(go.Scatter(
#                 x=x_positions,
#                 y=sev_subset['USUBJID'],
#                 mode='markers',
#                 marker=dict(color=severity_colors[predicted_severity], size=10, line=dict(width=1, color='DarkSlateGrey')),
#                 name=f'{endpoint} - {predicted_severity}',
#                 legendgroup=predicted_severity,
#                 showlegend=False if f'{endpoint} - {predicted_severity}' in [trace.name for trace in fig.data] else True,
#                 hoverinfo='x+y'
#             ))

#     # Add a trace for each severity category just for the legend
#     for sev_category in severity_order:
#         fig.add_trace(go.Scatter(
#             x=[None],
#             y=[None],
#             mode='markers',
#             marker=dict(color=severity_colors[sev_category], size=10, line=dict(width=1, color='DarkSlateGrey')),
#             name=f'{endpoint} - {sev_category}',
#             legendgroup=sev_category,
#             showlegend=True
#         ))

#     # Add vertical lines between each true severity category
#     for i in range(1, len(severity_order)):
#         fig.add_shape(type="line",
#                       x0=i - 0.5,
#                       y0=0,
#                       x1=i - 0.5,
#                       y1=1,
#                       xref='x',
#                       yref='paper',
#                       line=dict(color="black", width=1))

#     fig.update_layout(
#         title={
#             'text': f'Misclassification of {endpoint} compared to {gold_standard_endpoint}, Visit: {visit if visit else "All"}',
#             'y': 0.95,
#             'x': 0.5,
#             'xanchor': 'center',
#             'yanchor': 'top'
#         },
#         xaxis=dict(
#             title=f'True Severity Category ({gold_standard_endpoint})',
#             categoryorder='array',
#             categoryarray=severity_order,
#             tickvals=list(range(len(severity_order))),
#             ticktext=severity_order,
#             tickangle=-45,
#             ticks='outside',
#             tickmode='array',
#             ticklen=10,
#             tickcolor='black'
#         ),
#         yaxis=dict(title='Subjects', showgrid=True, zeroline=True, showline=True, gridcolor='lightgray'),
#         legend_title_text=f'{endpoint} Severity Category',
#         plot_bgcolor='white'
#     )

#     fig.show()


# # Visualization Report
# @skill
# def two_endpoints_visualization_report(df, endpoint1, endpoint2, gold_standard_endpoint, bySeverityCategory=False):
#      """
#     Creates a visualization report of multiple visualizations by comparing two endpoints.
    
#     Parameters
#     ----------
#     df : DataFrame
#         The original DataFrame containing the data.
#             'VISIT': Visit name (e.g. VISIT2)
#             'USUBJID': Unique subject ID
#             'digital_EP': Endpoint name (e.g. WASO, AHI, etc.)
#             'digital_EP_value': Endpoint value (some numeric value)
#             'digital_EP_severity_category': Severity category of the endpoint
#             'COHORT': Treatment group (e.g. Placebo/ Treatment)
#             'DEVICE': Device name (e.g. WatchPAT, PSG, etc.)
#     endpoint1 : str
#         The endpoint to analyze (e.g., WASO, AHI, etc.).
#     endpoint2 : str
#         The endpoint to analyze (e.g., WASO, AHI, etc.) and compare endpoint1 against.
#     gold_standard_endpoint : str
#         The gold standard endpoint to compare against.
#     visit : str, optional
#         The visit to filter the data (default is to use all visits).

#     Returns
#     ----------
#     None
#     """
    
#     change_from_baseline_plot(df, endpoint1)
#     change_from_baseline_plot(df, endpoint2)
#     severity_category_confusion_matrix(df, endpoint1, visit1='Screening', visit2=None)
#     severity_category_confusion_matrix(df, endpoint2, visit1='Screening', visit2=None)
#     bland_altman_plot(df, endpoint1, endpoint2, bySeverityCategory)
#     plot_endpoint_distribution(df, endpoint1, endpoint2, bySeverityCategory=False)
#     plot_correlation(df, endpoint1, endpoint2, bySeverityCategory=False)
    
#     if endpoint1 == gold_standard_endpoint:
#         categorized_strip_plot(df, endpoint2, gold_standard_endpoint, visit=None)
#     else:
#         categorized_strip_plot(df, endpoint1, gold_standard_endpoint, visit=None)

