query = "Detect anomalies in the data"
response = """
import pandas as pd

# find the anomalies
detected_anomalies = detect_anomalies(data, lag, num_anomalies, output,output_path)

result = { "type": "dataframe", "value": detected_anomalies }
"""

query1 = "Show me which anomalies occur during the day."
response1 = """
df_day = getDayDf(df)

result = { "type": "dataframe", "value": df_day}
"""

query2 = "Show me which anomalies occur during the night."
response2 = """
df_night = getNightDf(df)

result = { "type": "dataframe", "value": df_night}
"""

query3 = "How many anomalies occur during the day?"
response3 = """
count_day = numberDayDf(df)

result = { "type": "integer", "value": count_day}
"""

query4 = "How many anomalies occur during the night?"
response4 = """
count_night = numberNightDf(df)

result = { "type": "integer", "value": count_night}
"""

query5 = "Show me what the start time of the first anomaly is."
response5 = """
strTimestamp = strtTimeDf(df)

result = {"type": "string", "value": strTimestamp}
"""

query6 = "Are there more anomalies occuring in the day or night?"
response6 = """
num = dayVsNightDf(df)

result = { "type": "integer", "value": num}
"""

query7 = "What is the most frequent time of anomalies?"
response7 = """
formatted_time = freqTimeDf(df)

result = { "type": "string", "value": formatted_time}
"""

query8 = "Show me the value of the average score of the anomalies in the data"
response8 = """
result = { "type": "value", "value": avgAnomScore}
"""

query9 = "Show me a scatter plot of the anomalies in the data."
response9 = """
result = { "type": "graph", "value": plotAnom}
"""

query10 = "Show me which anomalies have a score that's greater than 2.849544"
response10 = """
sortedDf = numAnomScoreDf()
result = { "type": "dataframe", "value": sortedDf}
"""


def get_training_materials():
    queries=[query, query2, query3, query4, query5, query6, query7, query8, query9, query10]
    codes=[response, response2, response3, response4, response5, response6, response7, response8, response9, response10]
    return queries, codes
 
 
 