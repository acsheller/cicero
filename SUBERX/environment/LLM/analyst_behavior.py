import os
import pandas as pd


class AnalystBehaviorSimulator:
    """
    A class to simulate analyst behaviors interacting with news articles using an LLM.
    """
    
    def __init__(self, agent,history_file,impressions_file,analysts_file,behaviors_file):
        """
        Initialize the simulator with an LLM agent and tracking for analysts.

        Args:
            agent: The LLM agent configured to generate responses.
            history_file: Path to the history file.
            impressions_file: Path to the impressions file.
            analyst_file: Path to the synthetic analysts file.
            behaviors_file: The behaviors file for this dataset.
        """
        self.agent = agent
        self.analyst_data = {}  # Dictionary to store analyst-specific history and impressions
        self.history_file = history_file
        self.impressions_file = impressions_file
        self.analyst_file = analysts_file
        self.history_df = self.load_or_create_file(self.history_file, ["uid", "history"])
        self.impressions_df = self.load_or_create_file(self.impressions_file, ["impression_id", "uid", "news_id", "clicked"])
        self.analysts = self.load_or_create_file(self.analyst_file, ["uid", "name", "age", "gender", "primary_news_interest", 
        "secondary_news_interest", "job", "description"])


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
            df = pd.read_csv(file_path, sep="\t")
            if columns:
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

    def add_to_impressions(self, impression_id, user_id, news_id, clicked):
        """
        Add an impression to the impressions DataFrame.

        Args:
            impression_id (str): The unique ID for the impression.
            user_id (str): The ID of the analyst.
            news_id (str): The ID of the news article.
            clicked (int): Whether the article was clicked (1 for clicked, 0 for not clicked).
        """
        new_entry = {
            "impression_id": impression_id,
            "user_id": user_id,
            "news_id": news_id,
            "clicked": clicked
        }
        self.impressions_df = pd.concat([self.impressions_df, pd.DataFrame([new_entry])], ignore_index=True)
        self.save_file(self.impressions_file, self.impressions_df)

    def add_to_history(self, user_id, news_id):
        """
        Add a news article to the user's history.

        Args:
            user_id (str): The ID of the analyst.
            news_id (str): The ID of the news article.
        """
        # Check if the user already has a history entry
        if user_id in self.history_df["user_id"].values:
            # Append the new article to the existing history
            current_history = self.history_df.loc[self.history_df["user_id"] == user_id, "history"].values[0]
            updated_history = f"{current_history} {news_id}" if current_history else news_id
            self.history_df.loc[self.history_df["user_id"] == user_id, "history"] = updated_history
        else:
            # Add a new entry for the user
            new_entry = {"user_id": user_id, "history": news_id}
            self.history_df = pd.concat([self.history_df, pd.DataFrame([new_entry])], ignore_index=True)

        # Save the updated history
        self.save_history()


if __name__ == '__main__':
    MIND_type = "MINDsmall"
    data_path_base="/home/asheller/cicero/datasets/"
    data_path = data_path_base + MIND_type + "/"

    history_file = data_path + "history.tsv"
    impressions_file = data_path + "impressions.tsv"
    behaviors_file  = data_path + "train/behavors.tsv"
    analysts_file = data_path_base + "synthetic_analysts.csv"


    simulator = AnalystBehaviorSimulator("AGENT",history_file=history_file,impressions_file=impressions_file,analysts_file=analysts_file,behaviors_file=behaviors_file)

    print(f"maybe complete")