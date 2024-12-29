import time

# Start the timer
start_time = time.time()



import modin.pandas as pd
#import pandas as pd
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
from tqdm.asyncio import tqdm_asyncio
import os

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


ANALYST_AREAS = [
    "Global Politics",
    "Economics and Markets",
    "Science and Innovation",
    "Health and Medicine",
    "Climate and Environment",
    "Sports and Entertainment",
    "Technology and Startups",
    "Crime and Legal Affairs",
    "Business and Corporate Affairs",
    "Culture and Society",
    "Education and Academia",
    "Infrastructure and Urban Development",
    "Energy and Sustainability",
    "Defense and Security",
    "Art and Design",
    "Food and Agriculture",
    "Travel and Tourism",
    "Religion and Philosophy",
    "Consumer Trends and Retail",
    "Space Exploration",
    "Artificial Intelligence and Machine Learning",
    "Blockchain and Cryptocurrency",
    "Mental Health and Wellness",
    "Social Media and Digital Culture",
    "Activism and Social Justice",
    "Emerging Technologies",
    "Crisis Response"
]

class AnalystProfile(BaseModel):
    """
    This is the structure the LLM will return.
    """
    name: str = Field(description = 'A single unique name consisting of a first and last name.')
    age: int = Field(description = 'Age of the analyst', ge=25, le = 65)
    gender: str = Field(description= 'Gender: Male, Female, or Non-binary. With a distribution identicial to the real world population')
    primary_news_interest: str = Field(description= 'Primary catagory of news Interest')
    secondary_news_interest: str = Field(description= 'Secondary catagory of news Interest')
    job: str = Field(description= 'Job title  e.g. Technology Analyst')
    description: str = Field(description='The background of the analyst in their field of expertise')

    def __str__(self):
        return (
            f"AnalystProfile:\n"
            f"  Name: {self.name}\n"
            f"  Age: {self.age}\n"
            f"  Gender: {self.gender}\n"
            f"  Primary News Interest: {self.primary_news_interest}\n"
            f"  Secondary News Interest: {self.secondary_news_interest}\n"
            f"  Job: {self.job}\n"
            f"  Description: {self.description}\n"
        )
    
    def __repr__(self):
        return self.__str__()



used_names = set()

def validate_analyst_data(data: str) -> Optional[dict]:
    """
    Validate the generated JSON data for the synthetic analyst profile.

    Args:
        data (str): JSON string representing the synthetic analyst profile.

    Returns:
        Optional[dict]: Parsed and validated dictionary, or None if validation fails.
    """

    def is_valid_name(name: str) -> bool:
        """Validate if the name is in 'First Last' format."""
        return bool(re.fullmatch(r"[A-Z][a-z]+ [A-Z][a-z]+", name))

    def is_valid_gender(gender: str) -> bool:
        """Validate if gender is either 'M' or 'F'."""
        return gender in {"M", "F"}

    def is_valid_age(age: int) -> bool:
        """Validate if age is between 25 and 65."""
        return 25 <= age <= 65

    try:
        if isinstance(data, AnalystProfile):
            profile = data.model_dump()
        elif isinstance(data,str):
            profile = json.loads(data)
        required_fields = ["name", "age", "gender", "primary_news_interest", 
                           "secondary_news_interest", "job", "description"]


        if not all(field in profile for field in required_fields):
            print("Missing Required Fields.")
            return None

        # Additional validation of names
        if not is_valid_name(profile["name"]):
            print(f"Invalid name format: {profile['name']}")
            return None
        if profile["gender"] in ["Male", "MALE"]:
            profile["gender"] = "M"
        elif profile["gender"] in ["Female","FEMALE"]:
            profile["gender"] = "F"
        if not is_valid_gender(profile["gender"]):
            print(f"Invalid gender: {profile['gender']}. Must be 'M' or 'F'.")
            return None

        if not is_valid_age(profile["age"]):
            print(f"Invalid age: {profile['age']}. Must be between 25 and 65.")
            return None

        return profile

    except (json.JSONDecodeError, KeyError, ValueError):
        print("something went wrong in validate_analyst_data")
    return None


async def generate_single_profile(area: str, max_retries: int = 3) -> Optional[dict]:
    """Generate a single profile with retries and debug logging."""
    for attempt in range(max_retries + 1):
        try:
            result = await generate_synthetic_analyst_with_llm(area)
            
            if result:
                return result
            else:
                print("Profile validation failed or result was None.")
        except Exception as e:
            print(f"Exception during profile generation (attempt {attempt + 1}): {e}")
    return None



async def generate_synthetic_analyst_with_llm(analyst_area: str, retries: int = 3, delay: int = 2) -> Optional[dict]:
    """
    Generate a synthetic analyst profile for a given area of expertise using LLM.

    Args:
        analyst_area (str): The primary area of expertise for the synthetic analyst.
        retries (int): Number of retries in case of failure.
        delay (int): Delay in seconds between retries.

    Returns:
        Optional[dict]: A dictionary representing the synthetic analyst profile, or None if generation fails.
    """
    prompt = f"""
        Create a synthetic analyst profile as valid JSON. Do not include names or personal identifiers in the description. Use the following structure:
        {{
            "name": "string",                 # Unique full name, e.g., "Joe Smith".
            "age": int,                       # Integer between 25 and 65.
            "gender": "M" or "F",             # Use M for male, F for female.
            "primary_news_interest": "string", # '{analyst_area}'.
            "secondary_news_interest": "string", # Related to '{analyst_area}'.
            "job": "string",                  # Realistic job title.
            "description": "string"           # Detailed professional background, work habits, and news consumption. No names or personal identifiers.
        }}
        """
    for attempt in range(retries):
        try:
            profile = None
            result = await agent.run(prompt)
            if result:
                profile = validate_analyst_data(result.data)
            if profile:
                return profile
        except Exception as e:
            if attempt < retries - 1:
                await asyncio.sleep(delay * (2 ** attempt))  # Exponential backoff
            else:
                return None


agent = Agent(model=ollama_model, result_type=AnalystProfile, retries=5)


import os
import pandas as pd
from tqdm import tqdm

async def gen_analysts_profiles(num_profiles: int, analyst_areas: list, data_path_base: str, max_retries: int = 3) -> pd.DataFrame:
    """
    Generate additional synthetic analyst profiles with retry logic for errors and save progress to a CSV file.

    Args:
        num_profiles (int): Number of new profiles to generate.
        analyst_areas (list): List of primary news interests for analysts.
        data_path_base (str): Base path where the CSV file is located.
        max_retries (int): Maximum number of retries for a failed generation.

    Returns:
        pd.DataFrame: DataFrame containing all profiles, including the newly generated ones.
    """
    # Define file path
    file_path = os.path.join(data_path_base, "synthetic_analysts.csv")

    # Initialize used names set
    used_names = set()

    # Check if the file exists
    if os.path.exists(file_path):
        existing_data = pd.read_csv(file_path)
        print(f"Existing data loaded. {len(existing_data)} records found.")
        
        # Populate used names set with names from the existing data
        if "name" in existing_data.columns:
            used_names.update(existing_data["name"])
    else:
        existing_data = pd.DataFrame()
        print("No existing file found. Starting fresh.")

    # Calculate total profiles required
    total_existing_profiles = len(existing_data)
    total_required_profiles = total_existing_profiles + num_profiles

    profiles = []
    with tqdm(total=num_profiles, desc="Generating Synthetic Analysts") as pbar:
        while len(existing_data) + len(profiles) < total_required_profiles:
            area = analyst_areas[(len(profiles)+len(existing_data)) % len(analyst_areas)]
            retries = 0
            while retries <= max_retries:
                try:
                    # Attempt to generate the profile
                    profile = await generate_synthetic_analyst_with_llm(area)
                    if profile and profile["name"] not in used_names:
                        profiles.append(profile)
                        used_names.add(profile["name"])  # Add name to the used set
                        
                        pbar.update(1)
                        # Save progress every 10 profiles
                        if len(profiles) % 10 == 0 or total_existing_profiles + len(profiles) == total_required_profiles:
                            print(f"Saving progress after {len(profiles)} new profiles.")
                            new_data = pd.DataFrame(profiles)
                            existing_data = pd.concat([existing_data, new_data], ignore_index=True).drop_duplicates(subset="name",keep="first").reset_index(drop=True)
                            existing_data.to_csv(file_path, index=False)
                            print(f"Progress saved to {file_path}. Total records: {len(existing_data)}.")
                            profiles = []  # Clear the in-memory profiles list after saving
                        
                        # Check if the task is complete
                            if len(existing_data) + len(profiles) == total_required_profiles:
                                print(f"Task complete. Generated {num_profiles} new profiles.")
                                break


                except Exception as e:
                    retries += 1
                    if retries > max_retries:
                        print(f"Failed to generate profile for '{area}' after {max_retries} retries: {e}")
                        break  

    # Return the final DataFrame
    return pd.concat([existing_data, pd.DataFrame(profiles)], ignore_index=True)



async def main():
    
    analyst_area = random.choice(ANALYST_AREAS)
    result =  await generate_single_profile(analyst_area)
    print(f"result is {result}")


async def main2():
    num_profiles = 10
    profiles_df = await gen_analysts_profiles(2,ANALYST_AREAS,data_path_base)
    return profiles_df    

if __name__ == "__main__":
    df = asyncio.run(main2())