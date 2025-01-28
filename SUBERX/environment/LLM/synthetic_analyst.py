'''
Generate Synthetic Users 

'''

import os, re, json, random, uuid
import asyncio
import pandas as pd
from tqdm import tqdm
# Imports for Pydantic AI
from pydantic_ai import Agent
from pydantic import BaseModel, Field
from pydantic_ai.models.ollama import OllamaModel
from typing import Dict, Optional, List


# This should be mounted into the contaainer when it runs
# /app/datasets when running in container
data_path_base = "/home/acshell/cicero/datasets/"

model_name = "mistral:7b"  # Replace with your preferred model
#model_name = "llama3.2"  # Replace with your preferred model

ollama_url = "http://localhost:11434/v1/"  # Ollama's default base URL



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


class SimulatedAnalystGenerator:
    '''
    A class to geneate Simulated Users using LLMs
    '''

    def __init__(self,analysts_file,num_profiles):
        """
        Constructor

        """
        self.analysts_file = analyst_file
        if os.path.exists(analysts_file):
            self.analysts = pd.read_csv(self.analysts_file)
        else:
            self.analysts = pd.DataFrame()
        

        self.used_names = set()
        if "name" in self.analysts.columns:
            self.used_names.update(self.analysts['name'])


        total_existing_profiles = len(self.analysts)
        self.total_required_profiles = total_existing_profiles+ num_profiles
        self.connect_ollama(ollama_url=ollama_url,model_name=model_name)
        self.create_agent()
        

    def connect_ollama(self,ollama_url,model_name):
        '''
        This depends on an ollama server running serving-up models.

        '''

        self.ollama_model = OllamaModel(
        model_name=model_name,  
        base_url=ollama_url
        )


    def create_agent(self):
        """
        Create a pydantic AI agent
        """
        if not hasattr(self,'agent'):
            self.agent = Agent(model=self.ollama_model,result_type=AnalystProfile,retries=5)



    def validate_analyst_data(self, data: str) -> Optional[dict]:
        """
        Validate the generated JSON data for the synthetic analyst profile.

        Args:
            data (str): JSON string representing the synthetic analyst profile.

        Returns:
            Optional[dict]: Parsed and validated dictionary, or None if validation fails.
        """

        def is_valid_name(name: str) -> bool:
            """Validate if the name has at least a first and last name."""
            # Allow letters, hyphens, periods, and mixed case
            return bool(re.fullmatch(r"[A-Za-z.-]+ [A-Za-z.-]+", name))


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



    async def generate_single_profile_with_llm(self,analyst_area: str, retries: int = 3, delay: int = 2) -> Optional[dict]:
        '''
        Generate One Analyst Profile

        '''
        prompt = f"""
        Generate a synthetic analyst profile with the following structure:
        {{
            "name": "First Last",
            "age": integer (25-65),
            "gender": "M" or "F",
            "primary_news_interest": "{analyst_area}",
            "secondary_news_interest": "Related topic",
            "job": "Job title",
            "description": "Brief background"
        }}
        """


        for attempt in range(retries):
            try:
                profile = None
                result = await self.agent.run(prompt)
                if result:
                    profile = self.validate_analyst_data(result.data)
                if profile:
                    return profile
            except Exception as e:
                pass
                #print(f"Exception during profile generation (attempt {attempt + 1}:  {e})")
        return None



    async def generate_profiles(self,num_profiles: int = 10, max_retries: int =3) -> pd.DataFrame:
        '''
        Generate Analysts Profiles

        '''
        used_names = set()
        profiles = []
        with tqdm(total= num_profiles, desc="Generating Synthetic Analysts") as pbar:
            while len(self.analysts) +len(profiles) < self.total_required_profiles:
                area = ANALYST_AREAS[(len(profiles)+len(self.analysts)) % len(ANALYST_AREAS)]
                retries = 0
                while retries <= max_retries:
                    try:
                        profile = await self.generate_single_profile_with_llm(area)
                        if profile and profile['name'] not in used_names:
                            profiles.append(profile)
                            self.used_names.add(profile["name"])

                            pbar.update(1)

                            if len(profiles) % 10 == 0 or self.total_required_profiles == len(self.analysts) + len(profiles):
                            
                                print(f"Preparing to save Data. length of Analysts is: {len(self.analysts)}")
                                print(f"Length of Profiles is {len(profiles)}")
                                new_data = pd.DataFrame(profiles)
                                new_data["name"] = new_data["name"].str.strip()

                                # Check for duplicates
                                duplicate_profiles = new_data[new_data["name"].isin(self.analysts["name"])]
                                print(f"Duplicate profiles being removed: {len(duplicate_profiles)}")

                                self.analysts = pd.concat([self.analysts, new_data], ignore_index=True).drop_duplicates(subset="name",keep="first").reset_index(drop=True)
                                self.analysts.to_csv(self.analysts_file,index=False)

                                profiles = []
                                print(f"Length of Analysts is now: {len(self.analysts)}")
                                print(f"Length of Profiles is {len(profiles)}")
                                print(f"Total Required Profiles: {self.total_required_profiles}, Current Total: {len(self.analysts) + len(profiles)}")                               
                                if self.total_required_profiles == len(self.analysts) + len(profiles):
                                    break
                    except Exception as e:
                        print(f"Exception {e}")
                        retries += 1
                        if retries > max_retries:
                            break            
        return self.analysts


if __name__ == '__main__':


    analyst_file = data_path_base + "synthetic_analysts.csv"
    analyst_generator = SimulatedAnalystGenerator(analysts_file=analyst_file,num_profiles=691)
    try:
        asyncio.run(analyst_generator.generate_profiles(num_profiles=100))

    except Exception as e:
        print(f"some sort of exeception {e}")