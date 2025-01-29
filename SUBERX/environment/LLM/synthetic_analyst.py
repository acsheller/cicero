'''
Generate Synthetic Users 

'''

import os, re, json, random, uuid
import asyncio
import pandas as pd
from tqdm import tqdm
# Imports for Pydantic AI
from pydantic_ai import Agent
from pydantic import BaseModel, Field, field_validator
from pydantic_ai.models.ollama import OllamaModel
from typing import Dict, Optional, List, ClassVar


# This should be mounted into the contaainer when it runs
# /app/datasets when running in container
data_path_base = "/home/asheller/cicero/datasets/"

#model_name = "mistral:7b"  # Replace with your preferred model
model_name = "llama3.2"  # Replace with your preferred model

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


    # Validate name
    @field_validator("name")
    @classmethod
    def validate_name(cls, name):
        # Validate the name format (first and last name, optionally a middle name)
        if not re.fullmatch(r"[A-Za-z.-]+( [A-Za-z.-]+){1,2}", name):
            raise ValueError("Name must have at least a first and last name, optionally a middle name.")

        # Check for duplicates in the existing names set
        if name.strip() in cls.used_names:
            raise ValueError(f"Name '{name}' is already used.")

        return name.strip()  # Normalize the name (strip whitespace)


    # Validate age
    @field_validator("age")
    @classmethod
    def validate_age(cls, age):
        if not (25 <= age <= 65):
            raise ValueError("Age must be between 25 and 65.")
        return age

    # Validate gender
    @field_validator("gender")
    @classmethod
    def validate_gender(cls, gender):
        gender = gender.upper()
        if gender == 'MALE':
            gender = 'M'
        elif gender == "FEMALE":
            gender = 'F'
        if gender not in {"M", "F"}:
            raise ValueError("Gender must be 'M' or 'F'.")
        return gender


    @classmethod
    def set_used_names(cls, names: set):
        """Set the existing names from a set."""
        cls.used_names = names


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
        
        print(f'{len(self.analysts)} Analysts Profiles Exist')
        

        self.used_names = set()
        if "name" in self.analysts.columns:
            self.used_names.update(self.analysts['name'])
            AnalystProfile.set_used_names(self.used_names)

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


    def validate_analyst_data(self, data) -> Optional[dict]:
        """
        Validate the generated data for the synthetic analyst profile.

        Args:
            data (str | dict | AnalystProfile): The input data to validate.

        Returns:
            Optional[dict]: Parsed and validated dictionary, or None if validation fails.
        """
        try:
            # Handle different input types
            if isinstance(data, AnalystProfile):
                profile = data.model_dump()  # Extract validated dictionary from Pydantic instance
            elif isinstance(data, str):
                profile = json.loads(data)  # Parse JSON string into a dictionary
            elif isinstance(data, dict):
                profile = data  # Use the dictionary as-is
            else:
                raise ValueError("Input data must be a string, dictionary, or AnalystProfile instance.")

            # Ensure all required fields are present
            required_fields = ["name", "age", "gender", "primary_news_interest",
                            "secondary_news_interest", "job", "description"]
            if not all(field in profile for field in required_fields):
                print("Missing required fields.")
                return None

            # Validate using Pydantic model
            validated_profile = AnalystProfile(**profile)
            return validated_profile.model_dump()  # don't  use dict()

        except json.JSONDecodeError:
            print("Invalid JSON format.")
        except ValidationError as e:
            print(f"Validation error: {e}")
        except KeyError as e:
            print(f"Missing key: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

        return None


    async def generate_single_profile_with_llm(self,analyst_area: str, retries: int = 3, delay: int = 2) -> Optional[dict]:
        '''
        Generate One Analyst Profile

        '''
        prompt = f"""
        Generate a synthetic analyst profile with the following structure. Ensure that the name is unique, realistic, and not repeated across requests. Follow these constraints:

        {{
            "name": "A unique and realistic name in the format 'First Last' or optionally 'First Middle Last'. Avoid using names from previous outputs.",
            "age": An integer between 25 and 65,
            "gender": "M" or "F",
            "primary_news_interest": "{analyst_area}",
            "secondary_news_interest": "A related topic to {analyst_area}",
            "job": "A realistic job title relevant to the primary_news_interest",
            "description": "A brief professional background, unique to the individual"
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
                # Uncomment the below to see the exception.  it happens frequently. 
                # print(f"Exception during profile generation (attempt {attempt + 1}:  {e})")
        return None



    async def generate_profiles(self,num_profiles: int = 10, max_retries: int =3) -> pd.DataFrame:
        '''
        Generate Analysts Profiles

        '''
        
        profiles = []
        with tqdm(total= num_profiles, desc="Generating Synthetic Analysts") as pbar:
            while len(self.analysts) +len(profiles) < self.total_required_profiles:
                area = ANALYST_AREAS[(len(profiles)+len(self.analysts)) % len(ANALYST_AREAS)]
                retries = 0
                while retries <= max_retries:
                    try:
                        profile = await self.generate_single_profile_with_llm(area)
                        if profile and profile['name'] not in self.used_names:
                            profiles.append(profile)
                            self.used_names.add(profile["name"])
                            AnalystProfile.set_used_names(self.used_names)
                            pbar.update(1)

                            if len(profiles) % 5 == 0 or self.total_required_profiles == len(self.analysts) + len(profiles):
                            
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
    analyst_generator = SimulatedAnalystGenerator(analysts_file=analyst_file,num_profiles=305)
    try:
        asyncio.run(analyst_generator.generate_profiles(num_profiles=305))

    except Exception as e:
        print(f"some sort of exeception {e}")