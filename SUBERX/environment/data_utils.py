
import pandas as pd
from tqdm import tqdm
import os
import pickle
import numpy as np
from nltk.tokenize import word_tokenize


def generate_uid2index(behaviors_file, output_file="uid2index.pkl"):
    """
    Generate uid2index.pkl mapping user_id to integer indices from behaviors.tsv.
    
    Args:
        behaviors_file (str): Path to the behaviors.tsv file.
        output_file (str): Path to save the uid2index.pkl file.
    
    Returns:
        dict: Mapping of user_id to index.
    """
    # Load behaviors.tsv
    columns = ["impression_id", "user_id", "time", "history", "impressions"]
    behaviors_df = pd.read_csv(behaviors_file, sep="\t", names=columns)

    # Create a mapping of user_id to index
    user_ids = behaviors_df["user_id"].dropna().unique()  # Drop NaN and get unique users
    uid2index = {user_id: idx for idx, user_id in enumerate(user_ids)}

    # Save as uid2index.pkl
    with open(output_file, "wb") as f:
        pickle.dump(uid2index, f)

    print(f"Created {output_file} with {len(uid2index)} users.")
    return uid2index




def load_glove_embeddings(glove_file, embedding_dim=300):
    """
    Load GloVe embeddings into a dictionary.
    
    Args:
        glove_file (str): Path to the GloVe file.
        embedding_dim (int): Dimension of GloVe vectors.
    
    Returns:
        dict: Mapping of words to embedding vectors.
    """
    embeddings_index = {}
    with open(glove_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading GloVe embeddings"):
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs
    print(f"Loaded {len(embeddings_index)} word vectors.")
    return embeddings_index


def create_word_dict_and_embeddings(news_file, glove_embeddings, embedding_dim=300, output_dir="."):
    """
    Create word_dict.pkl and embedding.npy using GloVe embeddings.

    Args:
        news_file (str): Path to news.tsv.
        glove_embeddings (dict): Loaded GloVe embeddings.
        embedding_dim (int): Dimension of GloVe vectors.
        output_dir (str): Directory to save outputs.
    
    Returns:
        dict, np.ndarray: word_dict and embedding_matrix.
    """
    from collections import Counter
    
    # Load news data
    news_df = pd.read_csv(news_file, sep="\t", names=["news_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"])

    # Tokenize titles and abstracts
    def tokenize(text):
        return word_tokenize(text.lower())  # Use the recommenders utility function

    all_text = news_df["title"].fillna("") + " " + news_df["abstract"].fillna("")
    tokens = []
    for text in all_text:
        tokens.extend(tokenize(text))
    
    # Count word frequencies and create word_dict
    word_counter = Counter(tokens)
    word_dict = {word: idx for idx, (word, _) in enumerate(word_counter.items(), start=1)}  # Start index at 1
    
    # Create embedding matrix
    embedding_matrix = np.zeros((len(word_dict) + 1, embedding_dim))  # Extra row for padding (index 0)
    for word, idx in word_dict.items():
        if word in glove_embeddings:
            embedding_matrix[idx] = glove_embeddings[word]
        else:
            embedding_matrix[idx] = np.random.normal(size=(embedding_dim,))  # Random vector for unknown words
    
    # Save word_dict and embedding matrix
    word_dict_file = os.path.join(output_dir, "word_dict.pkl")
    embedding_file = os.path.join(output_dir, "embedding.npy")
    with open(word_dict_file, "wb") as f:
        pickle.dump(word_dict, f)
    np.save(embedding_file, embedding_matrix)

    print(f"Saved word_dict.pkl and embedding.npy to {output_dir}.")
    return word_dict, embedding_matrix

 