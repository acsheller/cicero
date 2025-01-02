import time

# Start the timer
start_time = time.time()


#import modin.pandas as pd
import pandas as pd
from collections import defaultdict
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from typing import Dict, Optional, List
import uuid
import random
import asyncio
from tqdm.notebook import tqdm
import re
import json

# Define the Ollama model running on Ollama
ollama_model = OllamaModel(
    model_name="mistral:7b-instruct",  # Replace with your preferred model  Could be 'mistrel:7b', 'granite3.1-dense:latest', 'llama3.2', gemma2
    base_url="http://ollama:11434/v1/"  # Ollama's default base URL
)


MIND_type = 'MINDsmall'

data_path_base="/app/datasets/"
data_path = data_path_base + MIND_type +"/"


#behaviors_file = data_path + "train/behaviors.tsv"
#print(f"Behaviors File {behaviors_file}")

news_file = data_path + "train/news.tsv"
news_df = pd.read_csv(news_file, sep="\t", names=["news_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"])
#print(f"News file {news_file}")
# Load the behaviors data
columns = ["impression_id", "user_id", "time", "history", "impressions"]
#behaviors_df = pd.read_csv(behaviors_file, sep="\t", names=columns)

def print_elapsed_time(start_time):
    """
    Print the elapsed time since `start_time` in hours, minutes, and seconds.
    
    Args:
        start_time (float): The starting time, typically obtained from time.time().
    """
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Elapsed Time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
