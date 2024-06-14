import pandas as pd
from setup_create_pandasai_agent import create_agent
import os

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

prompt = "Plot a correlation plot comparing AHI for WatchPat and PSG."
df = pd.read_csv("./data/standardized_analysis_ready_df.csv")
df.drop(columns=['Unnamed: 0'], inplace=True)
create_agent(df, prompt)
print("test done")