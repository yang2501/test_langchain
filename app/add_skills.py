import matplotlib.pyplot as plt
import pandas as pd
from pandasai import skill

# Plot bland_altman to compare device/ endpoint alignment

@skill
def bland_altman_plot(df, endpoint1, endpoint2, device1=None, device2=None, bySeverityCategory=False):
    """
    Generates a Bland-Altman plot to compare two devices or two endpoints, optionally by severity category.

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
    device1 : str, optional
        The first device to compare (e.g., WatchPAT).
    device2 : str, optional
        The second device to compare (e.g., PSG).
    bySeverityCategory : bool, optional
        Whether to plot the Bland-Altman plots by severity category.

    Returns
    ----------
    str
        Confirmation message after plotting.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    if endpoint1 == endpoint2:
        # Filter the DataFrame for the specified endpoint and devices
        df1 = df[(df['digital_EP'] == endpoint1) & (df['DEVICE'] == device1)][['USUBJID', 'VISIT', 'digital_EP_value', 'digital_EP_severity_category']].rename(columns={'digital_EP_value': f'digital_EP_value_{device1}'})
        df2 = df[(df['digital_EP'] == endpoint1) & (df['DEVICE'] == device2)][['USUBJID', 'VISIT', 'digital_EP_value', 'digital_EP_severity_category']].rename(columns={'digital_EP_value': f'digital_EP_value_{device2}'})
    else:
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
            if endpoint1 == endpoint2:
                category_data['mean'] = category_data[[f'digital_EP_value_{device1}', f'digital_EP_value_{device2}']].mean(axis=1)
                category_data['difference'] = category_data[f'digital_EP_value_{device1}'] - category_data[f'digital_EP_value_{device2}']
            else:
                category_data['mean'] = category_data[[f'{endpoint1}_value', f'{endpoint2}_value']].mean(axis=1)
                category_data['difference'] = category_data[f'{endpoint1}_value'] - category_data[f'{endpoint2}_value']

            # Calculate the mean difference and limits of agreement
            mean_diff = category_data['difference'].mean()
            std_diff = category_data['difference'].std()
            upper_limit = mean_diff + 1.96 * std_diff
            lower_limit = mean_diff - 1.96 * std_diff
            count = len(category_data['difference'])

            # Create the Bland-Altman plot
            plt.figure(figsize=(8, 6))
            plt.scatter(category_data['mean'], category_data['difference'], alpha=0.5)
            plt.axhline(mean_diff, color='gray', linestyle='solid', label=f'Mean Difference: {mean_diff:.2f}')
            plt.axhline(upper_limit, color='red', linestyle='--', label=f'+1.96 SD: {upper_limit:.2f}')
            plt.axhline(lower_limit, color='red', linestyle='--', label=f'-1.96 SD: {lower_limit:.2f}')
            plt.xlabel('Means')
            plt.ylabel(f'Difference: {endpoint1} - {endpoint2}' if endpoint1 != endpoint2 else f'Difference: {endpoint1} ({device1}) - {endpoint1} ({device2})')
            plt.title(f'Bland-Altman Plot for {count} {endpoint1} - {endpoint2} pairs\nSeverity Category: {category}' if endpoint1 != endpoint2 else f'Bland-Altman Plot for {count} {device1} - {device2} pairs\nSeverity Category: {category}')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            plt.savefig("../plt.png")
    else:
        # Calculate the mean and difference of the two endpoints
        if endpoint1 == endpoint2:
            df_merged['mean'] = df_merged[[f'digital_EP_value_{device1}', f'digital_EP_value_{device2}']].mean(axis=1)
            df_merged['difference'] = df_merged[f'digital_EP_value_{device1}'] - df_merged[f'digital_EP_value_{device2}']
        else:
            df_merged['mean'] = df_merged[[f'{endpoint1}_value', f'{endpoint2}_value']].mean(axis=1)
            df_merged['difference'] = df_merged[f'{endpoint1}_value'] - df_merged[f'{endpoint2}_value']

        # Calculate the mean difference and limits of agreement
        mean_diff = df_merged['difference'].mean()
        std_diff = df_merged['difference'].std()
        upper_limit = mean_diff + 1.96 * std_diff
        lower_limit = mean_diff - 1.96 * std_diff
        count = len(df_merged['difference'])

        # Create the Bland-Altman plot
        plt.figure(figsize=(8, 6))
        plt.scatter(df_merged['mean'], df_merged['difference'], alpha=0.5)
        plt.axhline(mean_diff, color='gray', linestyle='solid', label=f'Mean Difference: {mean_diff:.2f}')
        plt.axhline(upper_limit, color='red', linestyle='--', label=f'+1.96 SD: {upper_limit:.2f}')
        plt.axhline(lower_limit, color='red', linestyle='--', label=f'-1.96 SD: {lower_limit:.2f}')
        plt.xlabel('Means')
        plt.ylabel(f'Difference: {endpoint1} - {endpoint2}' if endpoint1 != endpoint2 else f'Difference: {endpoint1} ({device1}) - {endpoint1} ({device2})')
        plt.title(f'Bland-Altman Plot for {count} {endpoint1} - {endpoint2} pairs' if endpoint1 != endpoint2 else f'Bland-Altman Plot: for {count} {device1} - {device2} pairs')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        plt.savefig('./plot.png')
        

# Plot change from baseline. By default, if no device is specified, it plots the change from baseline for an endpoint for all devices.

@skill
def change_from_baseline_plot(df, endpoint, device=None):
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
    device : str, optional
        The device to plot. If not specified, the default is to make a change from baseline plot for all devices.

    Returns
    ----------
    None
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    
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

    def plot_device_data(df_filtered, endpoint, device):
        """
        Helper function to plot the data for a specific device.

        Parameters
        ----------
        df_filtered : DataFrame
            Filtered DataFrame containing the data for a specific device.
        endpoint : str
            The Digital_EP to plot (e.g., WASO, AHI, etc.).
        device : str
            The device to include in the plot title.

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
        title = f"{device} {endpoint} Change from Baseline"
        plt.title(title, fontsize=10)
        plt.xlabel('Visit', fontsize=10)
        plt.ylabel(f'{endpoint} Change', fontsize=10)
        plt.grid()
        plt.tight_layout()
        plt.show()
    
    # Filter by endpoint
    df_filtered = df[df['digital_EP'] == endpoint]
    
    # If device is specified, plot for that device
    if device:
        df_filtered = df_filtered[df_filtered['DEVICE'].str.lower() == device.lower()]
        plot_device_data(df_filtered, endpoint, device)
    else:
        # If no device is specified, plot for each device
        for dev in df_filtered['DEVICE'].unique():
            df_device = df_filtered[df_filtered['DEVICE'] == dev]
            print(f"Plotting for device: {dev}")
            plot_device_data(df_device, endpoint, dev)

# Plot endpoint distribution for each device for each visit to have a general idea of device agreement

@skill
def plot_endpoint_distribution(df, endpoint, device1=None, device2=None, visit=None, bySeverityCategory=False):
    """
    Plots histograms showing the distribution of a specified endpoint for each device and compares the means.
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
    endpoint : str
        The Digital_EP to plot (e.g., WASO, AHI, etc.).
    device1 : str, optional
        The first device to filter the data (e.g., WatchPAT, PSG).
    device2 : str, optional
        The second device to filter the data (e.g., WatchPAT, PSG).
    visit : str, optional
        The visit to filter the data (e.g., VISIT2). If not specified, the default is to plot for all visits where data for both devices is available.
    bySeverityCategory : bool, optional
        Whether to plot the distribution by severity category.

    Returns
    ----------
    None
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from scipy.stats import ttest_ind

    # Filter the DataFrame for the specified endpoint
    df_filtered = df[df['digital_EP'] == endpoint]

    # Filter by visit if specified
    if visit:
        df_filtered = df_filtered[df_filtered['VISIT'].str.lower() == visit.lower()]

    # Ensure both devices are specified
    if device1 is None or device2 is None:
        devices = df['DEVICE'].unique()
        if len(devices) < 2:
            print("Not enough devices found in the data.")
            return
        device1, device2 = devices[:2]

    # Include only patients with data from both devices
    subjects_with_both_devices = df_filtered[df_filtered['DEVICE'] == device1]['USUBJID'].isin(
        df_filtered[df_filtered['DEVICE'] == device2]['USUBJID'])
    common_subjects = df_filtered[df_filtered['DEVICE'] == device1][subjects_with_both_devices]['USUBJID']
    df_filtered = df_filtered[df_filtered['USUBJID'].isin(common_subjects)]

    if bySeverityCategory:
        # Get the unique severity categories
        severity_categories = df_filtered['digital_EP_severity_category'].unique()

        # Create a plot for each severity category
        for category in severity_categories:
            category_data = df_filtered[df_filtered['digital_EP_severity_category'] == category]

            # Create a histogram for each device within the severity category
            plt.figure(figsize=(12, 8))

            sns.histplot(data=category_data, x='digital_EP_value', hue='DEVICE', multiple='dodge', bins=20, kde=True)

            subject_count = category_data['USUBJID'].nunique()
            plt.title(f'{endpoint} Distribution by Device for Severity Category: {category} ({subject_count} subjects)')
            plt.xlabel(f'{endpoint} Value')
            plt.ylabel('Count')
            plt.legend(title='Device')
            plt.tight_layout()
            plt.show()
    else:
        # Identify visits where data for both devices is available
        visits = df_filtered['VISIT'].unique()
        valid_visits = []
        for v in visits:
            visit_data = df_filtered[df_filtered['VISIT'].str.lower() == v.lower()]
            if all(d in visit_data['DEVICE'].unique() for d in [device1, device2]):
                valid_visits.append(v)
        visits = valid_visits

        statistics = []

        # Create a plot for each visit
        for visit in visits:
            visit_data = df_filtered[df_filtered['VISIT'].str.lower() == visit.lower()]

            device_data1 = visit_data[visit_data['DEVICE'].str.lower() == device1.lower()]
            device_data2 = visit_data[visit_data['DEVICE'].str.lower() == device2.lower()]

            if device_data1.empty or device_data2.empty:
                print(f"No data found for endpoint '{endpoint}' in visit '{visit}' for devices '{device1}' and '{device2}'.")
                continue

            # Plot the histograms
            fig, axs = plt.subplots(1, 2, figsize=(20, 6))
            axs[0].hist(device_data1['digital_EP_value'], bins=20, alpha=0.7, color='blue', edgecolor='black', label=device1)
            axs[0].set_title(f'{endpoint} values for {device1} in {visit}', fontsize=15)
            axs[0].set_xlabel(f'{endpoint} Value', fontsize=12)
            axs[0].set_ylabel('Frequency', fontsize=12)
            axs[0].grid(axis='y', alpha=0.75)
            axs[0].legend()

            axs[1].hist(device_data2['digital_EP_value'], bins=20, alpha=0.7, color='green', edgecolor='black', label=device2)
            axs[1].set_title(f'{endpoint} values for {device2} in {visit}', fontsize=15)
            axs[1].set_xlabel(f'{endpoint} Value', fontsize=12)
            axs[1].set_ylabel('Frequency', fontsize=12)
            axs[1].grid(axis='y', alpha=0.75)
            axs[1].legend()

            plt.suptitle(f'Distribution of {endpoint} values in {visit}', fontsize=18)

            # Calculate the means and perform a t-test to compare means
            mean1 = device_data1['digital_EP_value'].mean()
            mean2 = device_data2['digital_EP_value'].mean()
            t_stat, p_value = ttest_ind(device_data1['digital_EP_value'], device_data2['digital_EP_value'])

            # Generate conclusion based on p-value
            if p_value < 0.05:
                conclusion = f'There is a statistically significant difference between {device1} and {device2} in {visit} (p < 0.05).'
            else:
                conclusion = f'There is no statistically significant difference between {device1} and {device2} in {visit} (p >= 0.05).'

            # Store statistics
            statistics.append({
                'Visit': visit,
                f'{device1} Mean': mean1,
                f'{device2} Mean': mean2,
                'p-value': p_value,
                'Conclusion': conclusion
            })

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

        # Create a DataFrame from the statistics and display it
        stats_df = pd.DataFrame(statistics)
        print(stats_df)
        for conclusion in stats_df['Conclusion']:
            print(conclusion)

# Plot endpoint correlation

@skill
def plot_correlation(df, endpoint1, endpoint2, device1=None, device2=None, bySeverityCategory=False):
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
    device1 : str, optional
        The first device to filter the data (e.g., WatchPAT, PSG).
    device2 : str, optional
        The second device to filter the data (e.g., WatchPAT, PSG).
    bySeverityCategory : bool, optional
        Whether to plot the correlation by severity category.

    Returns
    ----------
    None
    """
    
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from scipy.stats import linregress, spearmanr, pearsonr

    # Filter the DataFrame for the specified endpoints
    if endpoint1 == endpoint2:
        df1 = df[(df['digital_EP'] == endpoint1) & (df['DEVICE'] == device1)][['USUBJID', 'VISIT', 'digital_EP_value', 'digital_EP_severity_category', 'DEVICE']].rename(columns={'digital_EP_value': f'digital_EP_value_{device1}'})
        df2 = df[(df['digital_EP'] == endpoint1) & (df['DEVICE'] == device2)][['USUBJID', 'VISIT', 'digital_EP_value', 'digital_EP_severity_category', 'DEVICE']].rename(columns={'digital_EP_value': f'digital_EP_value_{device2}'})
    else:
        df1 = df[df['digital_EP'] == endpoint1][['USUBJID', 'VISIT', 'digital_EP_value', 'digital_EP_severity_category', 'DEVICE']].rename(columns={'digital_EP_value': f'{endpoint1}_value'})
        df2 = df[df['digital_EP'] == endpoint2][['USUBJID', 'VISIT', 'digital_EP_value', 'digital_EP_severity_category', 'DEVICE']].rename(columns={'digital_EP_value': f'{endpoint2}_value'})

    # Merge the two DataFrames on the subject ID and visit
    df_merged = pd.merge(df1, df2, on=['USUBJID', 'VISIT', 'digital_EP_severity_category'], suffixes=(f'_{device1}', f'_{device2}'))

    # Check if there is data to plot
    if df_merged.empty:
        print("No overlapping data between the specified endpoints/devices.")
        return "No overlapping data to plot."

    def create_plot(x, y, hue, title, xlabel, ylabel):
        # Calculate the correlation coefficients and line of best fit
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        line = slope * x + intercept
        spearman_corr, spearman_p_value = spearmanr(x, y)
        pearson_corr, pearson_p_value = pearsonr(x, y)

        # Create the scatter plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=x, y=y, hue=hue, palette='deep')
        plt.plot(x, line, color='red', label=f'Line of Best Fit: y={slope:.2f}x+{intercept:.2f}')
        plt.title(f'{title}\n'
                  f'Spearman Correlation: {spearman_corr:.2f}\n       '
                  f'Pearson Correlation: {pearson_corr:.2f}\n p-value: {pearson_p_value:.4f}')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    if bySeverityCategory:
        # Get the unique severity categories
        severity_categories = df_merged['digital_EP_severity_category'].unique()

        # Create a plot for each severity category
        for category in severity_categories:
            category_data = df_merged[df_merged['digital_EP_severity_category'] == category]

            if endpoint1 == endpoint2:
                x = category_data[f'digital_EP_value_{device1}']
                y = category_data[f'digital_EP_value_{device2}']
            else:
                x = category_data[f'{endpoint1}_value']
                y = category_data[f'{endpoint2}_value']

            create_plot(x, y, category_data[f'DEVICE_{device1}'],
                        f'{device1} {endpoint1} vs {device2} {endpoint2} Correlation\nSeverity Category: {category}',
                        f'{device1} {endpoint1} Value', f'{device2} {endpoint2} Value')
    else:
        if endpoint1 == endpoint2:
            x = df_merged[f'digital_EP_value_{device1}']
            y = df_merged[f'digital_EP_value_{device2}']
        else:
            x = df_merged[f'{endpoint1}_value']
            y = df_merged[f'{endpoint2}_value']

        create_plot(x, y, df_merged[f'DEVICE_{device1}'],
                    f'{device1} {endpoint1} vs {device2} {endpoint2} Correlation',
                    f'{device1} {endpoint1} Value', f'{device2} {endpoint2} Value')

# Plot confusion matrix by severity category

@skill
def severity_category_confusion_matrix(df, endpoint, device=None, visit1='Screening', visit2=None):
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
    device : str, optional
        The device to filter the data (e.g., WatchPAT, PSG). If not specified, a random device is chosen.
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
    
    if visit2 is None:
        # Get the latest valid visit
        visits = df['VISIT'].str.extract(r'(\d+)', expand=False).dropna().astype(int)
        latest_visit_num = visits.max()
        visit2 = f'VISIT{latest_visit_num}'
    
    # Filter the DataFrame for the specified endpoint
    df_filtered = df[df['digital_EP'] == endpoint]
    
    if device is None:
        device = np.random.choice(df_filtered['DEVICE'].unique())
    
    df_filtered = df_filtered[df_filtered['DEVICE'] == device]
    
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
        # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=severity_order, yticklabels=severity_order, cbar=False)
        ax = sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', xticklabels=severity_order, yticklabels=severity_order, cbar=False)
        plt.xlabel(f'Severity Category at {visit2}', )
        plt.ylabel(f'Severity Category at {visit1}')
        plt.title(f'Confusion Matrix of Severity Categories for {endpoint}\nDevice: {device}, Cohort: {cohort}')
        ax.xaxis.set_ticks_position('top')
        plt.xticks(rotation=45)
        plt.show()


# Categorized Strip Plot
@skill
def categorized_strip_plot(df, endpoint, gold_standard_device, visit=None):
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
    gold_standard_device : str
        The device to be used as the gold standard for classification (e.g., PSG).
    visit : str, optional
        The visit to filter the data (default is to use all visits).

    Returns
    ----------
    None
    """
    
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Define severity order
    severity_order = ['No', 'Mild', 'Moderate', 'Severe']
    
    # Filter the DataFrame for the specified endpoint and visit
    df_filtered = df[df['digital_EP'] == endpoint]
    
    if visit:
        df_filtered = df_filtered[df_filtered['VISIT'].str.lower() == visit.lower()]

    # Ensure device names are handled case-insensitively
    df_filtered['DEVICE'] = df_filtered['DEVICE'].str.lower()
    gold_standard_device = gold_standard_device.lower()

    # Create a pivot table to compare devices
    df_pivot = df_filtered.pivot_table(index='USUBJID', columns='DEVICE', values='digital_EP_severity_category', aggfunc='first')
    
    # Check if the gold standard device is in the DataFrame
    if gold_standard_device not in df_pivot.columns:
        print(f"The gold standard device '{gold_standard_device}' is not in the DataFrame.")
        return
    
    # Iterate over devices to create plots
    for device in df_pivot.columns:
        if device == gold_standard_device:
            continue
        
        # Create a DataFrame for plotting
        plot_df = df_pivot[[gold_standard_device, device]].dropna().reset_index()
        plot_df['Severity'] = plot_df[gold_standard_device]
        
        # Create the strip plot for the gold standard device
        plt.figure(figsize=(12, 8))
        sns.stripplot(x=gold_standard_device, y='USUBJID', data=plot_df, order=severity_order, palette='deep', size=8, alpha=0.6, jitter=False)
        
        # Create the strip plot for the current device, using hue to differentiate severity categories
        sns.stripplot(x=device, y='USUBJID', data=plot_df, order=severity_order, hue='Severity', size=6, alpha=0.6, dodge=True, palette='deep', jitter=True)
        
        plt.xlabel(f'True Severity Category ({gold_standard_device.upper()} {endpoint})')
        plt.ylabel('Subjects')
        plt.title(f'Misclassification for {device.upper()} {endpoint} compared to {gold_standard_device.upper()} {endpoint}, Visit: {visit if visit else "All"}')
        plt.legend(title=f'{device.upper()} {endpoint} Severity Category', bbox_to_anchor=(1.05, 1), loc='upper left', labels=severity_order, handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=10) for c in sns.color_palette('deep', len(severity_order))])
        plt.tight_layout()
        plt.show()

from pandasai import Agent 

def add_skills_to_agent(agent: Agent):

    if agent is None:
        raise ValueError("Agent is not initialized")
    
    agent = agent.add_skills(bland_altman_plot, change_from_baseline_plot, plot_endpoint_distribution, plot_correlation, severity_category_confusion_matrix, categorized_strip_plot)
    return agent


