import os
import random
import asyncio
import pandas as pd

# Imports for Pydantic AI
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from typing import Dict, Optional, List
from datetime import datetime
from tqdm import tqdm


data_path_base = "/home/asheller/cicero/datasets/"


class AnalystBehavior(BaseModel):
    """
    This is the structure the LLM will return for behaviors.
    """
    impression_id: str = Field(description="A unique session identifier for this behavior entry.")
    user_id: str = Field(description="The unique identifier of the analyst.")
    news_id: str = Field(description="The unique identifier of the news article.")
    clicked: str = Field(description="a 1 if clicked, a 0 if not clicked")

    def __str__(self):
        return (
            f"AnalystBehavior:\n"
            f" Impression ID: {self.impression_id}\n"
            f" User ID: {self.user_id}\n"
            f" News ID: {self.news_id}\n"
            f" Clicked: {self.clicked}")
    
    def __repr__(self):
        return self.__str__()

class AnalystBehaviorSimulator:
    """
    A class to simulate analyst behaviors interacting with news articles using an LLM.
    """
    
    def __init__(self, history_file,impressions_file,analysts_file,behaviors_file,news_file,uid_to_names):
        """
        Initialize the simulator with an LLM agent and tracking for analysts.

        Args:
            agent: The LLM agent configured to generate responses.
            history_file: Path to the history file.
            impressions_file: Path to the impressions file.
            analyst_file: Path to the synthetic analysts file.
            behaviors_file: The behaviors file for this dataset.
        """
        self.analyst_data = {}  # Dictionary to store analyst-specific history and impressions
        self.history_file = history_file
        self.impressions_file = impressions_file
        self.analysts_file = analysts_file
        self.behaviors_file = behaviors_file
        self.news_file = news_file
        self.uid_to_names_file = uid_to_names

        self.impressions = self.load_or_create_file(self.impressions_file, ["impression_id", "uid", "news_id", "clicked"])
        
        self.analysts = pd.read_csv(self.analysts_file)
        
        
        self.behaviors = pd.read_csv(behaviors_file, sep="\t", header=None)
        self.behaviors.columns = ["impression_id", "user_id", "time", "history", "impressions"]
        self.news = pd.read_csv(news_file, sep="\t", header=None)
        self.news.columns = ["news_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]
        self.uid_to_names = pd.read_csv(uid_to_names,sep='\t')

        if len(self.uid_to_names) != len(self.analysts):
            # The file is empyt so
            num_analysts = len(self.analysts)
            obfuscated_ids = [f"A{random.randint(1000, 9999)}" for _ in range(num_analysts)]
            zipped_data = list(zip(obfuscated_ids,self.analysts["name"]))
            self.uid_to_names = pd.DataFrame(zipped_data, columns=["uid", "name"])
            self.save_file(self.uid_to_names_file, self.uid_to_names)

        self.uid_and_analysts = self.uid_to_names.merge(self.analysts, on="name", how="inner")

        self.history = self.load_or_create_file(self.history_file, ["uid", "history"])
        if len(self.history) != len(self.analysts):
            self.history = pd.DataFrame({"uid": self.uid_to_names["uid"],  # Take UIDs from the merged DataFrame
            "history": [[] for _ in range(len(self.uid_to_names))]  # Start with empty lists
            })
        self.save_file(self.history_file,self.history)

        # Need to create a next impression ID to be used.abs
        max_impression_id = self.behaviors["impression_id"].max()
        self.current_impression_id = max_impression_id + 1



    def load_or_create_file(self, file_path, columns=None):
        """
        Load a file if it exists, otherwise create an empty one with specified columns.

        Args:
            file_path (str): The path to the file.
            columns (list or None): List of column names for the DataFrame. If None, columns will not be pre-defined.

        Returns:
            pd.DataFrame: The loaded or newly created DataFrame.
        """
        if os.path.exists(file_path):
            print(f"Loading file from {file_path}")
            df = pd.read_csv(file_path, sep="\t",header=None)
            if columns and len(df.columns) == 0:
                df.columns = columns
            return df
            
        else:
            print(f"File not found. Creating {file_path}")
            if columns is not None:
                # Create a DataFrame with specified columns
                df = pd.DataFrame(columns=columns)
            else:
                # Create an empty DataFrame without predefined columns
                df = pd.DataFrame()
            df.to_csv(file_path, sep="\t", index=False)
            return df

    def save_file(self, file_path, df):
        """
        Save a DataFrame to a file.

        Args:
            file_path (str): The path to the file.
            df (pd.DataFrame): The DataFrame to save.
        """
        df.to_csv(file_path, sep="\t", index=False)
        print(f"File saved to {file_path}")

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
        Creates the Pydantic AI Agent 

        """
        if self.ollama_model:
            self.agent = Agent(model=self.ollama_model,result_type=AnalystBehavior,retries=5)


    def pick_random_user(self):
        """
        Pick a random user from the uid_to_names DataFrame, 
        ensuring all users are iterated through at least once.
        
        Returns:
            str: The randomly selected user ID.
        """
        # Initialize the set to track selected UIDs if it doesn't exist
        if not hasattr(self, "_selected_uids"):
            self._selected_uids = set()

        # Get all user IDs
        all_uids = set(self.uid_to_names["uid"].tolist())

        # Find the available UIDs that have not been selected yet
        available_uids = all_uids - self._selected_uids

        # If all users have been iterated, reset the set
        if not available_uids:
            print("All users have been iterated. Resetting.")
            self._selected_uids.clear()
            available_uids = all_uids

        # Pick a random user from the available UIDs
        selected_uid = random.choice(list(available_uids))
        self._selected_uids.add(selected_uid)  # Track the selected UID
        return selected_uid

    def pick_random_news(self, num_articles=1):
        """
        Pick a specified number of random news articles from the news DataFrame.

        Add code to monitor news articles are implement some selection scheme. 
        
        Args:
            num_articles (int): Number of news articles to select.
        Returns:
            pd.DataFrame: A DataFrame with the selected news articles.
        """
        return self.news.sample(n=num_articles)

    def __make_history(self,uid):
        '''
        Part of the behavior is the analyists previous history.
        '''
        history = list(self.history[self.history['uid'] == uid]['history'])[0]
        if len(history) == 0:
            return_string = "You have reviewed no articles."
        else:
            for hs in history:
                news_id,review =hs.split('-')
                article = self.news[self.news['']]
            return_string = "You have previously interacted with the following articles:"

        return return_string
    


    async def behavior_as_the_analyst(self,analyst_uid, impression_id, article):
        '''
  
        '''
        gender = {"F":'female',"M":'male'}
        
        # Get the analyst based on UID
        analyst = self.uid_and_analysts[self.uid_and_analysts['uid'] == analyst_uid].to_dict(orient='records')[0]

        # Need to get the history of the user from the history dataframe.

        # history = self.__make_history(analyst_uid)

        # Build the prompt
        prompt = f"""
        You are {analyst['name']}, a {analyst['age']}-year-old {analyst['gender']} working as a {analyst['job']}.
        {analyst['description']}

        Session Details:
        - User ID: {analyst['uid']}
        - Impression ID: {impression_id}
        - News ID: {article['news_id']}
        The news article is titled: '{article['title']}'

        Would you click on this article? (Respond with 1 for clicked, 0 for not clicked)
        """
        
        # Query the LLM
        result = None
        result = await self.agent.run(prompt)
        return result.data




    async def behavior_as_third_party(self, analyst, impression_id, history, article):
        '''
        Generate the behavior as an analyst one article at a time
        '''
        # Build the prompt
        prompt = f"""
        You are an expert judge of human character and news consumption behavior.

        Session Details:
        - User ID: {analyst['uid']}
        - Impression ID: {impression['impression_id']}

        Here is the user's profile:
        - Name: {analyst['name']}
        - Age: {analyst['age']}
        - Gender: {analyst['gender']}
        - Primary News Interest: {analyst['primary_news_interest']}
        - Secondary News Interest: {analyst['secondary_news_interest']}
        - Job: {analyst['job']}
        - Description: {analyst['description']}

        The user has previously interacted with the following articles:
        {', '.join(history)}

        Evaluate the following article:
        - Title: '{impression['title']}'
        - News ID: {impression['news_id']}

        Would they click on this article? (Respond with 1 for clicked, 0 for not clicked)
        What is your reasoning for clicking or not clicking on the article?
        """
        # Query the LLM
        result = None
        #result = await agent.run(prompt)
        return result

    def update_history(self, uid, news_id):
        '''
        Given the UID update the history of the User
        '''
        row_index = self.history[self.history['uid'] == uid].index
        if not row_index.empty:
            current_history = self.history.loc[row_index[0], 'history']
            current_history.append(news_id)
            self.history.at[row_index[0], 'history'] = current_history
        else:
            print("UID not found in the DataFrame")
        i =1


async def main():
    MIND_type = "MINDsmall"
    
    data_path = data_path_base + MIND_type + "/"

    history_file = data_path + "history.tsv"
    impressions_file = data_path + "impressions.tsv"
    behaviors_file = data_path + "train/behaviors.tsv"
    analysts_file = data_path_base + "synthetic_analysts.csv"
    news_file = data_path + "train/news.tsv"
    uid_to_names = data_path + "uid_to_name.tsv"

    # Create a behavior_simulator
    behavior_simulator = AnalystBehaviorSimulator(
        history_file=history_file,
        impressions_file=impressions_file,
        analysts_file=analysts_file,
        behaviors_file=behaviors_file,
        news_file=news_file,
        uid_to_names=uid_to_names
    )

    # Connect to Ollama
    #model_name = "mistral:7b"  # Replace with your preferred model
    model_name = "llama3.2"  # Replace with your preferred model

    base_url = "http://localhost:11434/v1/"  # Ollama's default base URL

    # Make a connection to the local Ollama Server
    behavior_simulator.connect_ollama(ollama_url=base_url, model_name=model_name)

    # Create an Agent after Ollama connection is established
    behavior_simulator.create_agent()

    # Now iterate through all the users twice
    behaviors = []
    #for _ in range(len(behavior_simulator.uid_to_names) * 3):

    total_iterations = len(behavior_simulator.uid_to_names) * 3
    with tqdm(total=total_iterations, desc="Processing analysts", unit="iteration") as pbar:
        for _ in range(total_iterations):

            # 1. Get an analyst
            analyst_uid = behavior_simulator.pick_random_user()

            # 2. Select the news articles for the analyst to review
            session_articles = behavior_simulator.pick_random_news(num_articles=random.choice([1, 2, 3, 4, 5]))

            # 3. Get the next impression ID
            impression_id = behavior_simulator.current_impression_id

            # 4. Get the analyst's behavior
            impressions_this_session = []
            for article in session_articles.to_dict(orient='records'):
                result = await behavior_simulator.behavior_as_the_analyst(analyst_uid, impression_id, article)
                item = result.news_id + '-' + result.clicked
                if  int(result.clicked):
                    behavior_simulator.update_history(analyst_uid, article['news_id'])
                impressions_this_session.append(item)
                # repeat for the articles

            current_timestamp = datetime.now()
            # Format it 
            formatted_timestamp = current_timestamp.strftime("%m/%d/%Y %I:%M:%S %p")

            row_index = behavior_simulator.history[behavior_simulator.history['uid'] == analyst_uid].index
            history = behavior_simulator.history.loc[row_index[0], 'history']

            behavior = {
                'impression_id': int(impression_id),
                'user_id': analyst_uid,
                'time': formatted_timestamp,
                'history': history,
                'impressions': impressions_this_session
            }
            
            behaviors.append(behavior)

            # Update the progress bar
            pbar.update(1)


    new_behaviors = pd.DataFrame(behaviors)
    new_behaviors_file = data_path + 'analyst_behavior.csv'
    behavior_simulator.
    ("Processing complete")



if __name__ == '__main__':
    asyncio.run(main())