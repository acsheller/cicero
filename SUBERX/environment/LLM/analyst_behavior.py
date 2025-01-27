


import os
import random
import asyncio
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from datetime import datetime
from typing import Dict, Optional, List

# Imports for Pydantic AI
from pydantic_ai import Agent
from pydantic import BaseModel, Field
from pydantic_ai.models.ollama import OllamaModel



# Depending on if you are working in the container or not.  
# data_path_base = "/home/asheller/cicero/datasets/" Would be outside the container
#data_path_base = "/app/datasets/"

# It is currently set to
data_path_base = "/home/asheller/cicero/datasets/"

model_name = "mistral:7b"  # Replace with your preferred model
#model_name = "llama3.2"  # Replace with your preferred model
#model_name = "deepseek-r1:7b"  # Replace with your preferred model

base_url = "http://localhost:11434/v1/"  # Ollama's default base URL

# Override these as you wish




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
    
    def __init__(self, history_file,analysts_file,behaviors_file,news_file,uid_to_names):
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
        
        self.analysts_file = analysts_file
        self.behaviors_file = behaviors_file
        self.news_file = news_file
        self.uid_to_names_file = uid_to_names

        
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


        if os.path.exists(self.history_file):
            try:
                if os.path.getsize(self.history_file) > 0:
                    self.history = pd.read_csv(self.history_file)
                    print(f'Loaded existing history from {self.history_file}')
                else:
                    print(f'{self.history_file} is empty. Initializing a new dataframe')
                    self.history = pd.DataFrame()
            except Exception as e:
                print(f'Error reading {self.history_file}: {e}')
                self.history = pd.DataFrame()
            
        else:
            print(f'{self.history_file} does not exist. Creating a new one.')
            self.history = pd.DataFrame()



        if len(self.history) != len(self.analysts):
            # Need to reinistalizing it
            self.history = pd.DataFrame({"uid": self.uid_to_names["uid"],  # Take UIDs from the merged DataFrame
            "history": [[] for _ in range(len(self.uid_to_names))]  # Start with empty lists
            })

        # Need to create a next impression ID to be used.abs
        max_impression_id = self.behaviors["impression_id"].max()
        self.current_impression_id = max_impression_id + 1



    def save_file(self, file_path, df):
        """
        Save a DataFrame to a file.

        Args:
            file_path (str): The path to the file.
            df (pd.DataFrame): The DataFrame to save.
        """
        try:
            df.to_csv(file_path, sep="\t", index=False)
            print(f"File saved to {file_path}")
        except Exception as e:
            print(f"Error saving file {file_path}: {e}")

    def connect_ollama(self, ollama_url, model_name):
        try:
            self.ollama_model = OllamaModel(
                model_name=model_name,
                base_url=ollama_url
            )
            print(f"Connected to Ollama model: {model_name}")
        except Exception as e:
            print(f"Error connecting to Ollama server at {ollama_url} with model {model_name}: {e}")
            self.ollama_model = None

    def create_agent(self):
        try:
            if self.ollama_model:
                self.agent = Agent(model=self.ollama_model, result_type=AnalystBehavior, retries=5)
                print("Agent successfully created.")
            else:
                raise ValueError("Ollama model not initialized. Cannot create agent.")
        except Exception as e:
            print(f"Error creating agent: {e}")
            self.agent = None


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

        Not used but lef in.  In case history becomes a part of it.
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
    

    async def behavior_as_the_analyst(self, analyst_uid, impression_id, article):
        try:
            gender = {"F": 'female', "M": 'male'}
            analyst = self.uid_and_analysts[self.uid_and_analysts['uid'] == analyst_uid].to_dict(orient='records')[0]

            prompt = f"""
            You are {analyst['name']}, a {analyst['age']}-year-old {gender[analyst['gender']]} working as a {analyst['job']}.
            {analyst['description']}
            Session Details:
            - User ID: {analyst['uid']}
            - Impression ID: {impression_id}
            - News ID: {article['news_id']}
            The news article is titled: '{article['title']}'

            Would you click on this article? (Respond with 1 for clicked, 0 for not clicked)
            """
            result = await self.agent.run(prompt)
            return result.data
        except Exception as e:
            #print(f"Error generating behavior for analyst {analyst_uid} with article {article['news_id']}: {e}")
            return None


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


async def main(total_iterations = None):
    try:
        MIND_type = "MINDsmall"
        data_path = data_path_base + MIND_type + "/"
        history_file = data_path + "history.tsv"
        impressions_file = data_path + "impressions.tsv"
        behaviors_file = data_path + "train/behaviors.tsv"
        analysts_file = data_path_base + "synthetic_analysts.csv"
        news_file = data_path + "train/news.tsv"
        uid_to_names = data_path + "uid_to_name.tsv"

        # Initialize the behavior simulator
        behavior_simulator = AnalystBehaviorSimulator(
            history_file=history_file,
            analysts_file=analysts_file,
            behaviors_file=behaviors_file,
            news_file=news_file,
            uid_to_names=uid_to_names
        )

        # Connect to Ollama server and create the agent
        behavior_simulator.connect_ollama(ollama_url=base_url, model_name=model_name)
        behavior_simulator.create_agent()

        behaviors = []
        if not total_iterations:
            total_iterations = len(behavior_simulator.uid_to_names) * 3
        with tqdm(total=total_iterations, desc="Processing analysts", unit="iteration") as pbar:
            for _ in range(total_iterations):
                try:
                    # Step 1: Pick a random analyst
                    analyst_uid = behavior_simulator.pick_random_user()

                    # Step 2: Select news articles for the analyst to review
                    session_articles = behavior_simulator.pick_random_news(num_articles=random.choice([1, 2, 3, 4, 5]))

                    # Step 3: Get the next impression ID
                    impression_id = behavior_simulator.current_impression_id

                    # Step 4: Generate behavior for each article with retry logic
                    impressions_this_session = []
                    for article in session_articles.to_dict(orient='records'):
                        retry_attempts = 0
                        max_retries = 5
                        success = False

                        while not success and retry_attempts < max_retries:
                            try:
                                result = await behavior_simulator.behavior_as_the_analyst(analyst_uid, impression_id, article)

                                # Validate the result
                                if result and hasattr(result, 'news_id') and hasattr(result, 'clicked'):
                                    item = result.news_id + '-' + result.clicked
                                    if int(result.clicked):
                                        behavior_simulator.update_history(analyst_uid, article['news_id'])
                                    impressions_this_session.append(item)
                                    success = True
                                else:
                                    raise ValueError("Incomplete or invalid result returned by the agent.")
                            except Exception as e:
                                retry_attempts += 1
                                #print(f"Error retrieving behavior for article {article['news_id']}, attempt {retry_attempts}: {e}")

                        if not success:
                            print(f"Failed to retrieve behavior for article {article['news_id']} after {max_retries} attempts. Skipping.")

                    # Skip if no valid impressions were generated
                    if not impressions_this_session:
                        print(f"No valid impressions for analyst {analyst_uid}, skipping.")
                        continue

                    # Step 5: Create a behavior record
                    current_timestamp = datetime.now().strftime("%m/%d/%Y %I:%M:%S %p")
                    history = behavior_simulator.history.loc[
                        behavior_simulator.history['uid'] == analyst_uid, 'history'
                    ].iloc[0]

                    behavior = {
                        'impression_id': int(impression_id),
                        'user_id': analyst_uid,
                        'time': current_timestamp,
                        'history': history,
                        'impressions': impressions_this_session
                    }
                    behaviors.append(behavior)

                except Exception as e:
                    print(f"Error processing analyst behavior for UID {analyst_uid}: {e}")

                # Update the progress bar
                pbar.update(1)

                if len(behaviors) % 10 == 0 or len(behaviors) >= total_iterations:
                    # Define the file path for the CSV
                    new_behaviors_file = data_path + 'analyst_behavior.csv'

                    # Check if the file exists
                    if os.path.exists(new_behaviors_file):
                        try:
                            # Check if the file is not empty
                            if os.path.getsize(new_behaviors_file) > 0:
                                # Load the existing CSV into a DataFrame
                                existing_behaviors = pd.read_csv(new_behaviors_file)
                                #print(f"Loaded existing behaviors from {new_behaviors_file}")
                            else:
                                # File is empty, create an empty DataFrame
                                #print(f"{new_behaviors_file} is empty. Initializing a new DataFrame.")
                                existing_behaviors = pd.DataFrame()
                        except Exception as e:
                            print(f"Error reading {new_behaviors_file}: {e}")
                            existing_behaviors = pd.DataFrame()  # Create an empty DataFrame if reading fails
                    else:
                        # Create an empty DataFrame if the file doesn't exist
                        #print(f"{new_behaviors_file} does not exist. Creating a new one.")
                        existing_behaviors = pd.DataFrame()

                    # Convert current behaviors to a DataFrame
                    new_behaviors = pd.DataFrame(behaviors)

                    # Combine existing and new behaviors
                    combined_behaviors = pd.concat([existing_behaviors, new_behaviors], ignore_index=True)

                    # Save the combined DataFrame back to the CSV
                    try:
                        combined_behaviors.to_csv(new_behaviors_file, index=False)
                        #print(f"Saved combined behaviors to {new_behaviors_file}")
                        # TODO Double check this is working well
                        behavior_simulator.history.to_csv(behavior_simulator.history_file, index=False)
                    except Exception as e:
                        print(f"Error saving to {new_behaviors_file}: {e}")

                    # Reset the behaviors list
                    behaviors = []
 
 
        print("Processing complete")

    except Exception as e:
        print(f"Error in main execution: {e}")



if __name__ == '__main__':
    # With no arguments it will run for 3x the number of analysts.
    asyncio.run(main())