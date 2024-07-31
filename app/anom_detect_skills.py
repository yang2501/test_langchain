from pandasai import skill
import pandas as pd

@skill
def detect_anomalies(data, lag: int, num_anomalies = int, output = str, output_path = str):

    """
    Returns a dataframe with the start and end time of the detected anomaly, the score of the detected anomaly, and the level of the detected anomaly.
    Args:
        data: the data the look for anomalies in
        lag (int): the lag
        num_anomalies (int): the number of anomalies to find
    """
    def get_Anomalies(data, lag, num_anomalies, output, output_path):
        import bisect
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        
        def _compute_coef_matrix(w):
            from numpy import array, arange, ones, linalg, eye
            X = array([arange(w), ones([w])]).transpose()
            return X @ linalg.inv(X.transpose() @ X) @ X.transpose() - eye(w)
        
        def _partition_anomalies(windows, k):
            diffs = [windows[iw - 1][1] - windows[iw][1]
                     for iw in range(1, len(windows))]
            top_jump_positions = sorted(range(len(diffs)), key=lambda i: diffs[i], reverse=True)[0:k-1]
            return sorted(top_jump_positions) + [len(windows) - 1]
        
        def show_plot_raw(data):
        
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(data.index, data.values)
        
            plt.show()
            
        def show_plot(data, anomalies, thresholds):
            from matplotlib.patches import Rectangle
            import matplotlib.dates as mdates
        
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(data.index, data.values)
            plt.title('lag: {0}, #levels: {1}, #anomalies: {2}'
                            .format(1440, 5, 20))
        
            for anomaly in anomalies.values:  
                        start = mdates.date2num(pd.to_datetime(anomaly[1]))
                        end = mdates.date2num(pd.to_datetime(anomaly[2]))
                        width = end - start
                        height = ax.get_ylim()[1] - ax.get_ylim()[0]
                        rect = Rectangle((start, ax.get_ylim()[0]), width, height, color=plt.cm.jet(0.65 + float(anomaly[0]) / 5 / 3), alpha=0.5)
                        ax.add_patch(rect)
            plt.show()
        
        def detect_anomalies(data, lag, num_anomalies, num_levels=5, visualize=True,view_anomaly_table=True,view_plot=0):
            if type(data) != pd.Series:
                raise ValueError('data must be of the pandas Series type')
            if lag < 3:
                raise ValueError('lag needs to be at least 3.')
            if num_anomalies < 0:
                raise ValueError('expected number of anomalies must be positive.')
            num_levels = min(num_levels, num_anomalies) 
        
            data = data.fillna(method='pad')
        
            coefs = _compute_coef_matrix(lag)
        
            values = data.values
            num_windows = len(values) - lag + 1
            windows = np.vstack([values[ix:ix + num_windows] for ix in range(lag)])
            residuals = np.linalg.norm(coefs @ windows, axis=0)
        
            windows = [(ix, residuals[ix]) for ix in range(num_windows)]
            windows.sort(key=lambda item: item[1],
                         reverse=True)
        
            if num_anomalies == 0 or num_levels == 0:
                max_anomaly_score = windows[0][1]
                return None, [max_anomaly_score * 2]
        
            iw = 0
            top_iws = [iw]
            while len(top_iws) < num_anomalies:
                while iw < num_windows and any(abs(windows[jw][0] - windows[iw][0]) < lag for jw in top_iws):
                    iw += 1
                if iw < num_windows:
                    top_iws.append(iw)
                    iw += 1
                else:
                    break
            results = [windows[iw] for iw in top_iws]
        
            partition_points = _partition_anomalies(results, num_levels)
            thresholds = [results[iw][1] - 1e-3 for iw in partition_points]
        
            timestamps = data.index
            anomalies = []
            rank = 0
            for level, limit in enumerate(partition_points):
                while rank <= limit:
                    iw = results[rank][0]
                    anomalies.append((num_levels - level, str(timestamps[iw]), str(timestamps[iw + lag - 1]), results[rank][1]))
                    rank += 1
            anomalies = pd.DataFrame(anomalies, columns=['level', 'start', 'end', 'score'])
            anomalies = anomalies.sort_values(['score'], ascending=[False])
        
            if visualize:
                data.plot(title='lag: {0}, #levels: {1}, #anomalies: {2}'
                          .format(lag, num_levels, num_anomalies))
                for anomaly in anomalies.values:
                    plt.axvspan(pd.to_datetime(anomaly[1]), pd.to_datetime(anomaly[2]), color=plt.cm.jet(0.65 + float(anomaly[0]) / num_levels / 3), alpha=0.5)
            return anomalies, thresholds
        
        def anomalies_to_series(anomalies, index):
            rows = anomalies.shape[0]
            series = pd.Series(np.zeros(len(index)), dtype=np.int)
            series.index = index
            for r in range(rows):
                start = anomalies.loc[r, 'start']
                end = anomalies.loc[r, 'end']
                level = int(anomalies.loc[r, 'level'])
                series[start:end] = level
            return series
        
        class StreamingAnomalyDetector:
            def __init__(self, lag, thresholds):
                # This is prototype code and doesn't validate arguments
                self._w = lag
                self._thresholds = thresholds
                self._buffer = np.array([float('nan')] * lag)
                self._buffer.shape = (lag, 1)  # make it vertical
                self._coef_matrix = _compute_coef_matrix(lag)
        
            # Update thresholds on demand without restarting the service
            def update_thresholds(self, thresholds):
                self._thresholds = thresholds
        
            def score(self, value):
                self._buffer[:-1] = self._buffer[1:]
                self._buffer[-1] = value
                return np.linalg.norm(self._coef_matrix @ self._buffer)
        
            def classify(self, value):
                return bisect.bisect_left(self._thresholds, self.score(self, value))
        
        def anomaly_detector(data,lag, num_anomalies,output,output_path):
            anomalies, thresholds = detect_anomalies(data, lag=lag, num_anomalies=num_anomalies, visualize=False,
                                            view_anomaly_table=False,view_plot=0)
            if output == None:
                return anomalies
        
        def retrieveAnomalies(data,lag, num_anomalies,output,output_path):
            detected_anomalies = anomaly_detector(data,lag, num_anomalies,output,output_path)
            return detected_anomalies
            
        def getAnomalies(data, lag, num_anomalies, output,output_path):
            detected_anomalies = retrieveAnomalies(data,lag=lag, num_anomalies=num_anomalies, output=output, output_path = output_path)
        
            return detected_anomalies
    
    
        detected_anomalies = getAnomalies(data, lag, num_anomalies, output,output_path)
        return detected_anomalies
    
    detected_anomalies = get_Anomalies(data, lag, num_anomalies, None, None)
    return detected_anomalies

@skill
def getDayDf(df):
    '''
    Detects the anomalies that occurs in the day. 

    Parameters: 
    df: the dataFrame containing the data of the anomalies
    '''
    df['start']= pd.to_datetime(df['start'])
    df['stripped_hour'] = df['start'].dt.strftime('%H')
    df['stripped_hour'] = df['stripped_hour'].astype(int)
    df_day = df.loc[(df['stripped_hour'] >= 6) & (df['stripped_hour'] < 20)]
    return df_day 
  
@skill
def getNightDf(df):
    '''
    Detects the anomalies that occurs in the night.

    Parameters: 
    df: the dataFrame containing the data of the anomalies
    '''
    df['start']= pd.to_datetime(df['start'])
    df['stripped_hour'] = df['start'].dt.strftime('%H')
    df['stripped_hour'] = df['stripped_hour'].astype(int)
    df_night = df.loc[(df['stripped_hour'] >= 20) | (df['stripped_hour'] < 6)]
    return df_night

@skill
def numberDayDf(df):
    '''
    Finds the number of anomalies that occur in the day.

    Parameters: 
    df: the dataFrame containing the data of the anomalies
    '''
    df['start']= pd.to_datetime(df['start'])
    df['stripped_hour'] = df['start'].dt.strftime('%H')
    df['stripped_hour'] = df['stripped_hour'].astype(int)
    df_day = df.loc[(df['stripped_hour'] >= 6) & (df['stripped_hour'] < 20)]
    count_day = df_day.shape[0]

    return count_day
  
@skill
def numberNightDf(df):
    '''
    Finds the number of anomalies that occur in the night.

    Parameters: 
    df: the dataFrame containing the data of the anomalies
    '''
    df['start']= pd.to_datetime(df['start'])
    df['stripped_hour'] = df['start'].dt.strftime('%H')
    df['stripped_hour'] = df['stripped_hour'].astype(int)
    df_night = df.loc[(df['stripped_hour'] >= 20) | (df['stripped_hour'] < 6)]
    count_night = df_night.shape[0]

    return count_night

@skill
def strtTimeDf(df):
    '''
    Finds the start time of the first anomaly.

    Parameters: 
    df: the dataFrame containing the data of the anomalies
    '''
    df['start']= pd.to_datetime(df['start'])
    sorteddf = df.sort_values(['start'], ascending=[True])
    timestamp = sorteddf.iloc[0,1] 
    strTimestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    return strTimestamp

@skill 
def dayVsNightDf(df):
    '''
    Compares the number of anomalies that occurs during the day versus the night.

    Parameters: 
    df: the dataFrame containing the data of the anomalies
    '''
    df['start']= pd.to_datetime(df['start'])
    df['stripped_hour'] = df['start'].dt.strftime('%H')
    df['stripped_hour'] = df['stripped_hour'].astype(int)
    df_day = df.loc[(df['stripped_hour'] >= 6) & (df['stripped_hour'] < 20)]
    df_night = df.loc[(df['stripped_hour'] >= 20) | (df['stripped_hour'] < 6)]
    
    count_day = df_day.shape[0]
    count_night = df_night.shape[0]
    if count_day > count_night:
        #print ("There are more anomalies during the day.")
        return "There are more anomalies during the day."
    elif count_night > count_day:
        #print ("There are more anomalies during the night.")
        return "There are more anomalies during the night."
    else:
        #print ("There are the same number of anomalies during the day and night.")
        return "There are the same number of anomalies during the day and night."

@skill
def freqTimeDf(df):
    '''
    Finds the time with the most amount of anomalies.

    Parameters:
    df: the dataFrame containing the data of the anomalies
    hour: the time in military time that needs to be converted to standard time
    '''
    def format_hour(hour):
        from datetime import datetime
        dt = datetime.strptime(f"{hour}:00", "%H:%M")
        return dt.strftime("%I:%M %p")

    anom_frequent_time = df['stripped_hour'].value_counts().idxmax()
    count_frequent_time = df['stripped_hour'].value_counts().max()

    formatted_time = format_hour(anom_frequent_time)
    return formatted_time
  
@skill
def avgScoreDf(df):
    '''
    Finds the average value of the score of the anomalies in the data. 

    Parameters: 
    df: the dataFrame containing the data of the anomaly scores
    '''
    avgAnomScore = df['score'].mean()
    return avgAnomScore

@skill
def plotDf(df):
    '''
    Plots the anomalies in the dataFrame

    Parameters: 
    df: the dataFrame containing the data of the anomalies
    '''
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    df['start']= pd.to_datetime(df['start'])
    df['stripped_hour'] = df['start'].dt.strftime('%H')
    df['stripped_hour'] = df['stripped_hour'].astype(int)
    df["timeOfDay"] = ["day" if 6 <= hour < 20 else "night" for hour in df["stripped_hour"]]

    plotAnom = sns.relplot(
        data = df,
        x = "stripped_hour",
        y = "score",
        kind = "scatter",
        hue = "timeOfDay",
        palette ={"day": "blue", "night": "green"}
    )
    plotAnom.set(xlabel ="Hour", ylabel = "Score", title ='Score vs Hour')
    return plotAnom
  
@skill 
def numAnomScoreDf(df, sign, score):
    """
    Finds which anomalies fit parameters in relation to the score.

    Parameters:
    df: the dataFrame containing the data of the anomalies
    sign: the operator of the user's choosing
    score: the score of the user's choosing
    """
    import operator
    
    operators = {
        "<": operator.lt,
        ">": operator.gt,
        "<=": operator.le,
        ">=": operator.ge,
        "==": operator.eq,
        "!=": operator.ne
    }
    
    sortedDf = df.loc[operators[sign](df["score"], score)]
    countNumAnom = sortedDf.shape[0]
    if countNumAnom == 0:
        print("No anomalies match.")
        sortedDf = pd.DataFrame()
    else:
        print(countNumAnom, "anomalies", "\n")
    
    return sortedDf