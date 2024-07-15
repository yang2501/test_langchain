from pandasai import skill
import streamlit as st

# Plot bland_altman to compare endpoint alignment

@skill
def bland_altman_plot(df, endpoint1, endpoint2, bySeverityCategory=False):
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
    str
        Confirmation message after plotting.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import streamlit as st
    
    device1 = df.loc[df['digital_EP'] == endpoint1, 'DEVICE']
    device2 = df.loc[df['digital_EP'] == endpoint2, 'DEVICE']
    
    def plot_bland_altman(df, bySeverityCategory):
        # Calculate the mean difference and limits of agreement
            mean_diff = df['difference'].mean()
            std_diff = df['difference'].std()
            upper_limit = mean_diff + 1.96 * std_diff
            lower_limit = mean_diff - 1.96 * std_diff
            count = len(df['difference'])

            # Create the Bland-Altman plot
            plt.figure(figsize=(8, 6))
            plt.scatter(df['mean'], df['difference'], alpha=0.5)
            plt.axhline(mean_diff, color='gray', linestyle='solid', label=f'Mean Difference: {mean_diff:.2f}')
            plt.axhline(upper_limit, color='red', linestyle='--', label=f'+1.96 SD: {upper_limit:.2f}')
            plt.axhline(lower_limit, color='red', linestyle='--', label=f'-1.96 SD: {lower_limit:.2f}')
            plt.xlabel('Means')
            plt.ylabel(f'Difference: {endpoint1} - {endpoint2}' if endpoint1 != endpoint2 else f'Difference: {endpoint1} ({device1}) - {endpoint1} ({device2})')
            if bySeverityCategory:
                plt.title(f'Bland-Altman Plot for {count} {endpoint1} - {endpoint2} pairs\nSeverity Category: {category}' if endpoint1 != endpoint2 else f'Bland-Altman Plot for {count} {device1} - {device2} pairs\nSeverity Category: {category}')
            else:
                plt.title(f'Bland-Altman Plot for {count} {endpoint1} - {endpoint2} pairs' if endpoint1 != endpoint2 else f'Bland-Altman Plot: for {count} {device1} - {device2} pairs')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            fig = plt.gcf()
            st.pyplot(fig)

    # Filter the DataFrame for the specified endpoints
    df1 = df[(df['digital_EP'] == endpoint1)][['USUBJID', 'VISIT', 'digital_EP_value', 'digital_EP_severity_category']].rename(columns={'digital_EP_value': f'{endpoint1}_value'})
    df2 = df[(df['digital_EP'] == endpoint2)][['USUBJID', 'VISIT', 'digital_EP_value', 'digital_EP_severity_category']].rename(columns={'digital_EP_value': f'{endpoint2}_value'})

    # Merge the two DataFrames on the subject ID and visit
    df_merged = pd.merge(df1, df2, on=['USUBJID', 'VISIT', 'digital_EP_severity_category'])

    # Check if there is data to plot
    if df_merged.empty:
        print("No overlapping data between the specified endpoints/devices.")
        return "No overlapping data to plot."

    if bySeverityCategory:
        # Get the unique severity categories
        severity_categories = df_merged['digital_EP_severity_category'].unique()

        # Create a plot for each severity category
        for category in severity_categories:
            category_data = df_merged[df_merged['digital_EP_severity_category'] == category]

            # Calculate the mean and difference of the two endpoints
            category_data['mean'] = category_data[[f'{endpoint1}_value', f'{endpoint2}_value']].mean(axis=1)
            category_data['difference'] = category_data[f'{endpoint1}_value'] - category_data[f'{endpoint2}_value']
                
            plot_bland_altman(category_data, bySeverityCategory)

    else:
        # Calculate the mean and difference of the two endpoints
        df_merged['mean'] = df_merged[[f'{endpoint1}_value', f'{endpoint2}_value']].mean(axis=1)
        df_merged['difference'] = df_merged[f'{endpoint1}_value'] - df_merged[f'{endpoint2}_value']

        plot_bland_altman(df_merged, bySeverityCategory)

# Plot change from baseline. By default, if no device is specified, it plots the change from baseline for an endpoint for all devices.

@skill
def change_from_baseline_plot(df, endpoint):
    """
    Plots a change from baseline chart for different cohorts.

    Parameters
    ----------
    df : DataFrame
        The original DataFrame containing the data.
            'VISIT' (e.g. VISIT3). The get_visit_number() function relies on screening visit being marked as 'Screening'. The data should be cleaned such that there are only valid visits in this column
            'USUBJID': unique subject ID
            'digital_EP': (e.g. WASO, AHI, etc.)
            'digital_EP_value': (some numeric value)
            'digital_EP_severity_category' 
            'COHORT' (e.g. Placebo/ Treatment)
            'DEVICE' (WatchPAT, PSG, etc.)
    endpoint : str
        The Digital_EP to plot (e.g., WASO, AHI, etc.). If the endpoint is not provided. Ask the user a clarifying question for the endpoint.

    Returns
    ----------
    None
    """
    
    import matplotlib.pyplot as plt
    import streamlit as st

    def get_visit_number(visit):
        """
        Extracts the numeric part of the visit name for sorting.
        Non-numeric visits are considered invalid and return None.
        """
        import re
        
        match = re.match(r'^VISIT(\d+)$', visit, re.IGNORECASE)
        if match:
            return int(match.group(1))
        elif visit.lower() == 'screening':
            return 0
        else:
            return None

    def plot_device_data(df_filtered, endpoint):
        """
        Helper function to plot the data for a specific device.

        Parameters
        ----------
        df_filtered : DataFrame
            Filtered DataFrame containing the data for a specific device.
        endpoint : str
            The Digital_EP to plot (e.g., WASO, AHI, etc.).

        Returns
        ----------
        None
        """
        
        # Filter out invalid visits
        df_filtered['visit_num'] = df_filtered['VISIT'].apply(get_visit_number)
        df_filtered = df_filtered.dropna(subset=['visit_num'])

        # Ensure visit_num is integer type
        df_filtered['visit_num'] = df_filtered['visit_num'].astype(int)

        # Determine the visit with the highest postfix
        if not df_filtered['visit_num'].empty:
            max_visit_row = df_filtered.loc[df_filtered['visit_num'].idxmax()]
            comparison_visit = max_visit_row['VISIT']
        else:
            raise ValueError("No valid visits found.")

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

        fig = plt.figure(figsize=(8, 6))

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

            plt.errorbar(x, y, yerr=yerr, capsize=5, fmt='-o', color=cohort2Color[cohort], label=cohort, markersize=4)

        xticks = list(set(xticks))
        xticks.sort()

        unique_xlabels = []
        unique_xticks = []
        for tick, label in zip(xticks, xlabels):
            if tick not in unique_xticks:
                unique_xticks.append(tick)
                unique_xlabels.append(label)

        plt.xticks(unique_xticks, unique_xlabels, rotation=45)
        plt.legend(loc='best', prop={'size': 8})
        device = df.loc[df['digital_EP'] == endpoint, 'DEVICE']
        title = f"{device} {endpoint} Change from Baseline"
        plt.title(title, fontsize=10)
        plt.xlabel('Visit', fontsize=10)
        plt.ylabel(f'{endpoint} Change', fontsize=10)
        plt.grid()
        plt.tight_layout()
        fig = plt.gcf()
        st.pyplot(fig)
    
    # Filter by endpoint
    df_filtered = df[df['digital_EP'] == endpoint]
    
    # If device is specified, plot for that device
    plot_device_data(df_filtered, endpoint)


# Plot endpoint distribution for each device for each visit to have a general idea of device agreement

@skill
def plot_endpoint_distribution(df, endpoint1, endpoint2, bySeverityCategory=False):
    """
    Plots histograms showing the distribution of specified endpoints for each device and compares the means.
    Optionally, plots the distribution by severity category.

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
        The first endpoint to plot (e.g., WASO, AHI, etc.).
    endpoint2 : str
        The second endpoint to plot (e.g., WASO, AHI, etc.).
    bySeverityCategory : bool, optional
        Whether to plot the distribution by severity category.

    Returns
    ----------
    None
    """
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from scipy.stats import ttest_ind
    import streamlit as st
    
    def get_visit_number(visit):
        """
        Extracts the numeric part of the visit name for sorting.
        Non-numeric visits are considered invalid and return None.
        """
        import re
        match = re.match(r'^VISIT(\d+)$', visit, re.IGNORECASE)
        if match:
            return int(match.group(1))
        elif visit.lower() == 'screening':
            return 0
        else:
            return None

    # Filter the DataFrame for the specified endpoints
    df_endpoint1 = df[df['digital_EP'] == endpoint1]
    df_endpoint2 = df[df['digital_EP'] == endpoint2]

    if bySeverityCategory:
        # Get the unique severity categories
        severity_categories = df['digital_EP_severity_category'].unique()
        series = pd.Series(severity_categories)

        # Drop NaN values
        clean_series = series.dropna()

        # Convert back to NumPy array if needed
        severity_categories = clean_series.values

        for category in severity_categories:
            category_data1 = df_endpoint1[df_endpoint1['digital_EP_severity_category'] == category]
            category_data2 = df_endpoint2[df_endpoint2['digital_EP_severity_category'] == category]

            plt.figure(figsize=(12, 8))
            sns.histplot(data=category_data1, x='digital_EP_value', bins=20, kde=True, color='blue', label=endpoint1)
            sns.histplot(data=category_data2, x='digital_EP_value', bins=20, kde=True, color='green', label=endpoint2, alpha=0.5)

            subject_count1 = category_data1['USUBJID'].nunique()
            subject_count2 = category_data2['USUBJID'].nunique()
            plt.title(f'Distribution of {endpoint1} and {endpoint2} by Severity Category: {category} ({subject_count1} and {subject_count2} subjects)')
            plt.xlabel('Endpoint Value')
            plt.ylabel('Count')
            plt.legend()
            plt.tight_layout()
            fig = plt.gcf()
            st.pyplot(fig)
    else:
        # Filter out invalid visits
        df_endpoint1['visit_num'] = df_endpoint1['VISIT'].apply(get_visit_number)
        df_endpoint2['visit_num'] = df_endpoint2['VISIT'].apply(get_visit_number)
        df_endpoint1 = df_endpoint1.dropna(subset=['visit_num'])
        df_endpoint2 = df_endpoint2.dropna(subset=['visit_num'])

        # Ensure visit_num is integer type
        df_endpoint1['visit_num'] = df_endpoint1['visit_num'].astype(int)
        df_endpoint2['visit_num'] = df_endpoint2['visit_num'].astype(int)

        # Identify visits where data for both endpoints is available
        visits = set(df_endpoint1['VISIT'].unique()).intersection(df_endpoint2['VISIT'].unique())
        statistics = []

        for visit in visits:
            visit_data1 = df_endpoint1[df_endpoint1['VISIT'].str.lower() == visit.lower()]
            visit_data2 = df_endpoint2[df_endpoint2['VISIT'].str.lower() == visit.lower()]

            if visit_data1.empty or visit_data2.empty:
                continue

            fig, axs = plt.subplots(1, 2, figsize=(16, 6))

            sns.histplot(data=visit_data1, x='digital_EP_value', bins=20, kde=True, ax=axs[0], color='blue')
            sns.histplot(data=visit_data2, x='digital_EP_value', bins=20, kde=True, ax=axs[1], color='green')

            axs[0].set_title(f'{endpoint1} values in {visit}', fontsize=15)
            axs[1].set_title(f'{endpoint2} values in {visit}', fontsize=15)
            axs[0].set_xlabel(f'{endpoint1} Value', fontsize=12)
            axs[1].set_xlabel(f'{endpoint2} Value', fontsize=12)
            axs[0].set_ylabel('Frequency', fontsize=12)
            axs[1].set_ylabel('Frequency', fontsize=12)
            axs[0].grid(axis='y', alpha=0.75)
            axs[1].grid(axis='y', alpha=0.75)

            plt.suptitle(f'Distribution of {endpoint1} and {endpoint2} values in {visit}', fontsize=18)

            mean1 = visit_data1['digital_EP_value'].mean()
            mean2 = visit_data2['digital_EP_value'].mean()
            t_stat, p_value = ttest_ind(visit_data1['digital_EP_value'], visit_data2['digital_EP_value'])

            if p_value < 0.05:
                conclusion = f'There is a statistically significant difference between {endpoint1} and {endpoint2} in {visit} (p < 0.05).'
            else:
                conclusion = f'There is no statistically significant difference between {endpoint1} and {endpoint2} in {visit} (p >= 0.05).'

            statistics.append({
                'Visit': visit,
                f'{endpoint1} Mean': mean1,
                f'{endpoint2} Mean': mean2,
                'p-value': p_value,
                'Conclusion': conclusion
            })

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig = plt.gcf()
            st.pyplot(fig)

        # Create a DataFrame from the statistics and display it
        stats_df = pd.DataFrame(statistics)
        print(stats_df)
        for conclusion in stats_df['Conclusion']:
            print(conclusion)

# Plot endpoint correlation

@skill
def plot_correlation(df, endpoint1, endpoint2, bySeverityCategory=False):
    """
    Plots scatter plots showing the correlation between two endpoints for each device.
    Optionally, plots the correlation by severity category.

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
        Whether to plot the correlation by severity category.

    Returns
    ----------
    None
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from scipy.stats import linregress, spearmanr, pearsonr
    import streamlit as st

    # Filter the DataFrame for the specified endpoints
    df_endpoint1 = df[df['digital_EP'] == endpoint1]
    df_endpoint2 = df[df['digital_EP'] == endpoint2]

    # Merge the data for both endpoints to include only subjects with data for both
    common_subjects = pd.merge(
        df_endpoint1[['USUBJID', 'VISIT', 'digital_EP_value', 'digital_EP_severity_category']],
        df_endpoint2[['USUBJID', 'VISIT', 'digital_EP_value', 'digital_EP_severity_category']],
        on=['USUBJID', 'VISIT', 'digital_EP_severity_category'],
        suffixes=(f'_{endpoint1}', f'_{endpoint2}')
    )

    # Check if there is data to plot
    if common_subjects.empty:
        print("No overlapping data between the specified endpoints/devices.")
        return "No overlapping data to plot."

    def create_plot(x, y, title, xlabel, ylabel):
        # Calculate the correlation coefficients and line of best fit
        slope, intercept, r_value, p_value, std_err = linregress(x, y, alternative="two-sided")
        line = slope * x + intercept
        spearman_corr, spearman_p_value = spearmanr(x, y)
        pearson_corr, pearson_p_value = pearsonr(x, y)

        # Create the scatter plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=x, y=y)
        plt.plot(x, line, color='red', label=f'Line of Best Fit: y={slope:.2f}x+{intercept:.2f}')
        plt.title(f'{title}\n'
                  f'Spearman Correlation: {spearman_corr:.2f}\n       '
                  f'Pearson Correlation: {pearson_corr:.2f}\n p-value: {pearson_p_value:.4f}')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        fig = plt.gcf()
        st.pyplot(fig)

    if bySeverityCategory:
        # Get the unique severity categories
        severity_categories = common_subjects['digital_EP_severity_category'].unique()

        # Create a plot for each severity category
        for category in severity_categories:
            category_data = common_subjects[common_subjects['digital_EP_severity_category'] == category]

            x = category_data[f'digital_EP_value_{endpoint1}']
            y = category_data[f'digital_EP_value_{endpoint2}']

            create_plot(x, y,
                        f'{endpoint1} vs {endpoint2} Correlation\nSeverity Category: {category}',
                        f'{endpoint1} Value', f'{endpoint2} Value')
    else:
        x = common_subjects[f'digital_EP_value_{endpoint1}']
        y = common_subjects[f'digital_EP_value_{endpoint2}']

        create_plot(x, y,
                    f'{endpoint1} vs {endpoint2} Correlation',
                    f'{endpoint1} Value', f'{endpoint2} Value')


# Plot confusion matrix by severity category

# @skill
def severity_category_confusion_matrix(df, endpoint, visit1='Screening', visit2=None):
    """
    Generates confusion matrices for severity categories of an endpoint between two visits for each treatment cohort.

    Parameters
    ----------
    df : DataFrame
        The original DataFrame containing the data.
            'VISIT': Visit name (e.g. VISIT2)
            'USUBJID': Unique subject ID
            'digital_EP': Endpoint name (e.g. WASO, AHI, etc.)
            'digital_EP_severity_category': Severity category of the endpoint
            'COHORT': Treatment group (e.g. Placebo/ Treatment)
            'DEVICE': Device name (e.g. WatchPAT, PSG, etc.)
    endpoint : str
        The endpoint to analyze (e.g., WASO, AHI, etc.).
    visit1 : str, optional
        The first visit to compare (default is 'Screening').
    visit2 : str, optional
        The second visit to compare (default is the latest valid visit).

    Returns
    ----------
    None
    """
    
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    # Determine the latest visit if visit2 is not specified
    if visit2 is None:
        visits = df['VISIT'].str.extract(r'(\d+)', expand=False).dropna().astype(int)
        latest_visit_num = visits.max()
        visit2 = f'VISIT{latest_visit_num}'

    # Filter the DataFrame for the specified endpoint
    df_filtered = df[df['digital_EP'] == endpoint]

    # Get the unique treatment cohorts
    cohorts = df_filtered['COHORT'].unique()

    # Define the order of severity categories
    severity_order = ['Severe', 'Moderate', 'Mild', 'No']

    for cohort in cohorts:
        df_cohort = df_filtered[df_filtered['COHORT'] == cohort]

        # Filter data for visit1 and visit2
        df_visit1 = df_cohort[df_cohort['VISIT'].str.lower() == visit1.lower()][['USUBJID', 'digital_EP_severity_category']]
        df_visit2 = df_cohort[df_cohort['VISIT'].str.lower() == visit2.lower()][['USUBJID', 'digital_EP_severity_category']]

        # Rename columns to avoid confusion when merging
        df_visit1 = df_visit1.rename(columns={'digital_EP_severity_category': f'severity_category_{visit1}'})
        df_visit2 = df_visit2.rename(columns={'digital_EP_severity_category': f'severity_category_{visit2}'})

        # Merge the two visits on USUBJID
        df_merged = pd.merge(df_visit1, df_visit2, on='USUBJID')

        if df_merged.empty:
            print(f"No overlapping data between {visit1} and {visit2} for cohort {cohort}.")
            continue

        # Generate the confusion matrix
        y_true = pd.Categorical(df_merged[f'severity_category_{visit1}'], categories=severity_order, ordered=True)
        y_pred = pd.Categorical(df_merged[f'severity_category_{visit2}'], categories=severity_order, ordered=True)

        cm = pd.crosstab(y_true, y_pred, rownames=[f'Severity Category at {visit1}'], colnames=[f'Severity Category at {visit2}'], dropna=False)

        if cm.empty:
            print(f"No data to plot for cohort {cohort}.")
            continue

        # Calculate percentages
        cm_percentage = cm.div(cm.sum(axis=1), axis=0) * 100

        # Create annotations with counts and percentages
        annot = cm.astype(str) + "\n" + cm_percentage.round(2).astype(str) + '%'

        # Plot the confusion matrix
        plt.figure(figsize=(10, 7))
        ax = sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', xticklabels=severity_order, yticklabels=severity_order, cbar=False)
        plt.xlabel(f'Severity Category at {visit2}')
        plt.ylabel(f'Severity Category at {visit1}')
        plt.title(f'Confusion Matrix of Severity Categories for {endpoint}\nCohort: {cohort}')
        ax.xaxis.set_ticks_position('top')
        plt.xticks(rotation=45)
        fig = plt.gcf()
        st.pyplot(fig)


# Categorized Strip Plot
@skill
def categorized_strip_plot(df, endpoint, gold_standard_endpoint, visit=None):
    """
    Creates categorized strip plots for each device to visualize incorrect classifications compared to a gold standard device.

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
    endpoint : str
        The endpoint to analyze (e.g., WASO, AHI, etc.).
    gold_standard_endpoint : str
        The gold standard endpoint to compare against.
    visit : str, optional
        The visit to filter the data (default is to use all visits).

    Returns
    ----------
    None
    """
    
    import seaborn as sns
    import matplotlib.pyplot as plt
    import streamlit as st
    
    # Define severity order
    severity_order = ['No', 'Mild', 'Moderate', 'Severe']
    
    # Filter the DataFrame for the specified endpoint and visit
    df_filtered = df[(df['digital_EP'] == endpoint) | (df['digital_EP'] == gold_standard_endpoint)]
    
    if visit:
        df_filtered = df_filtered[df_filtered['VISIT'].str.lower() == visit.lower()]

    # Pivot the DataFrame to compare endpoints
    df_pivot = df_filtered.pivot_table(index='USUBJID', columns='digital_EP', values='digital_EP_severity_category', aggfunc='first')

    # Check if both endpoints are in the DataFrame
    if endpoint not in df_pivot.columns or gold_standard_endpoint not in df_pivot.columns:
        print(f"Both {endpoint} and {gold_standard_endpoint} must be present in the DataFrame.")
        return
    
    # Create a DataFrame for plotting
    plot_df = df_pivot[[gold_standard_endpoint, endpoint]].dropna().reset_index()
    plot_df['Severity'] = plot_df[gold_standard_endpoint]
    
    # Create the strip plot for the gold standard endpoint
    plt.figure(figsize=(12, 8))
    sns.stripplot(x=gold_standard_endpoint, y='USUBJID', data=plot_df, order=severity_order, palette='deep', size=8, alpha=0.6, jitter=False)
    
    # Create the strip plot for the specified endpoint, using hue to differentiate severity categories
    sns.stripplot(x=endpoint, y='USUBJID', data=plot_df, order=severity_order, hue='Severity', size=6, alpha=0.6, dodge=True, palette='deep', jitter=True)
    
    plt.xlabel(f'True Severity Category ({gold_standard_endpoint})')
    plt.ylabel('Subjects')
    plt.title(f'Misclassification of {endpoint} compared to {gold_standard_endpoint}, Visit: {visit if visit else "All"}')
    plt.legend(title=f'{endpoint} Severity Category', bbox_to_anchor=(1.05, 1), loc='upper left', labels=severity_order, handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=10) for c in sns.color_palette('deep', len(severity_order))])
    plt.tight_layout()
    fig = plt.gcf()
    st.pyplot(fig)
