import os
import numpy as np
import pandas as pd
import csv
from dataclasses import dataclass
from typing import List, Dict
import time
import ast

# Define paths to the MIND large dataset
## This is the original dataset path
#base_path_small = os.path.expanduser('~/efs/resources/datasets/mind_small_recsys_splits/')

## This is a temporary dataset path.  TODO Implement config file with this in it.
base_path_small = os.path.expanduser('/app/SUBERX/datasets/MINDsmall/')
train_path_small = os.path.join(base_path_small, 'train')
dev_path_small = os.path.join(base_path_small, 'validation_dev')
test_path_small = os.path.join(base_path_small, 'test')

# Paths to the embedding files
entity_train_embedding_path_small = os.path.join(train_path_small, 'entity_embedding.vec')
relation_train_embedding_path_small = os.path.join(train_path_small, 'relation_embedding.vec')
entity_dev_embedding_path_small = os.path.join(dev_path_small, 'entity_embedding.vec')
relation_dev_embedding_path_small = os.path.join(dev_path_small, 'relation_embedding.vec')
entity_test_embedding_path_small = os.path.join(test_path_small, 'entity_embedding.vec')
relation_test_embedding_path_small = os.path.join(test_path_small, 'relation_embedding.vec')


# Define column names for the datasets
news_columns_small = ['id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities']
behaviors_columns_small = ['impression_id', 'user_id', 'time', 'history', 'impressions']


def load_data_small(file_path, column_names):
    '''
    Function to load data into a Dataframe
    '''
    #print("---file_path is {}".format(file_path))
    return pd.read_csv(file_path_small, sep='\t', names=column_names)


def load_embeddings_small(path):
    # Read the embeddings file with pandas
    df = pd.read_csv(path, sep='\t', header=None)
    # TODO Review Wikidata and consider removing remove.
    # Assuming the first column is the WikidataId and the rest are embeddings
    wikidata_ids = df.iloc[:, 0].values
    embeddings = df.iloc[:, 1:].values
    return wikidata_ids, embeddings
    
# Function to get the embedding for a specific entity in the dictionary
def get_entity_embedding_small(news_article, entity_embeddings):
    entity_id = int(news_article['entity'])  # Assuming 'entity' column holds the entity ID
    return entity_embeddings[entity_id]

# Function to get the nearest neighbors
def search_entity_embedding_small(query_embedding, index, k=5):
    D, I = index.search(query_embedding, k)
    return D, I

# Function to extract the first entity ID from a column
def extract_first_entity_id_small(entities_column):
    if entities_column:
        entities_list = ast.literal_eval(entities_column)
        if entities_list:
            return entities_list[0]['WikidataId']
    return None

def print_duration_small(start_time, end_time):
    # Calculate the duration
    duration = end_time - start_time
    
    # Convert the duration to hours, minutes, and seconds
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = duration % 60
    
    # Print the formatted duration
    print(f"Process duration: {hours} hours, {minutes} minutes, and {seconds:.2f} seconds")

def get_wikidata_item_info_small(item_id):
    '''
    Function leveraging the wikidata API
    '''
    
    # Initialize the client
    client = wikidata.client.Client()

    # Fetch the item
    item = client.get(item_id, load=True)

    # Prepare the output
    item_info = {
        "Item ID": item.id,
        "Label": item.label,
        "Description": item.description,
        "Statements": []
    }

    # Print stuff
    for prop_id, value in item.data['claims'].items():
        prop_info = {"Property ID": prop_id, "Values": []}
        for statement in value:
            mainsnak = statement['mainsnak']
            if mainsnak['datatype'] == 'wikibase-item':
                entity_id = mainsnak['datavalue']['value']['id']
                entity = client.get(entity_id, load=True)
                prop_info["Values"].append(f"{entity.label} ({entity_id})")
            elif mainsnak['datatype'] == 'string':
                prop_info["Values"].append(mainsnak['datavalue']['value'])
            # Add more datatype handlers as needed
        item_info["Statements"].append(prop_info)

    return item_info

def save_catagories_to_csv_small(df):

    news_cats_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "../../mind/datasets/news_cats_small.csv",
    )

    unique_news_categories_list = df['category'].unique().tolist()
    unique_news_subcategories_list = df['subcategory'].unique().tolist()
    unique_combined_list = list(set(unique_news_categories_list + unique_news_subcategories_list))

    with open(news_cats_path, 'w', newline='') as file:
        writer = csv.writer(file)
        # If you want each item on a new row
        for item in unique_combined_list:
            writer.writerow([item])
    print("--- catagories save to file")
        # If you want all items in a single row, uncomment the line below and comment the loop above
        # writer.writerow(my_list)