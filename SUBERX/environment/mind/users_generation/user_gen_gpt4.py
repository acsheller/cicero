import os
import re
import csv
import openai
import pandas as pd
import numpy as np
import torch
import argparse
import config
import random 

class UserGenerator:
    def __init__(self, api_key):
        openai.api_key = api_key
        self.rng = np.random.default_rng()
        self.interests = pd.read_csv(config.INTERESTS_PATH)
        self.jobs = pd.read_csv(config.JOBS_PATH)

    def seed(self, seed):
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

    def generate_age(self, num):
        return self.rng.choice(
            a=[18, 25, 35, 45, 55, 65, 75],
            size=num,
            p=[0.15, 0.2, 0.2, 0.15, 0.15, 0.1, 0.05],
        ) + self.rng.integers(low=0, high=10, size=num)

    def sample_data(self, num, df, column):
        return df.sample(n=num, replace=True, random_state=self.rng)[column].values

    def generate_field(self, prompt, max_tokens=10):
        response = openai.chat.completions.create(
            model="gpt-4",  # or "gpt-3.5-turbo"
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()

    def generate_description(self, prompt):
        return self.generate_field(prompt, max_tokens=300)


    def parse_description(self, description):
        # Use regular expressions to find the fields
        name_match = re.search(r'Name:\s*(.*)', description)
        age_match = re.search(r'Age:\s*(\d+)', description)
        gender_match = re.search(r'Gender:\s*(\w+)', description)
        primary_interest_match = re.search(r'Primary News Interest:\s*(.*)', description)
        secondary_interest_match = re.search(r'Secondary News Interest:\s*(.*)', description)
        job_match = re.search(r'Job:\s*(.*)', description)
        full_description_match = re.search(r'Description:\s*(.*)', description, re.DOTALL)

        # Extract the matched groups
        name = name_match.group(1) if name_match else None
        age = int(age_match.group(1)) if age_match else None
        gender = gender_match.group(1) if gender_match else None
        primary_interest = primary_interest_match.group(1) if primary_interest_match else None
        secondary_interest = secondary_interest_match.group(1) if secondary_interest_match else None
        job = job_match.group(1) if job_match else None
        full_description = full_description_match.group(1).strip() if full_description_match else None

        # Return the parsed fields as a dictionary
        return {
            'name': name,
            'age': age,
            'gender': gender,
            'primary_news_interest': primary_interest,
            'secondary_news_interest': secondary_interest,
            'job': job,
            'description': full_description
        }


    def generate_persona(self, seed, num):
        ages = self.generate_age(num)
        interests = self.sample_data(num, self.interests, "Interest")
        primary_interest, secondary_interest = random.sample(self.interests['Interest'].tolist(), 2)
        jobs = self.sample_data(num, self.jobs, "Jobs")

        for i in range(len(ages)):
            if ages[i] > 65:
                jobs[i] = "retired"

        personas = []
        for i in range(num):
            name_prompt = "Generate a unique first and last name for a persona interested in news."
            gender_prompt = "Generate a gender (M/F) for a persona interested in news."
            description_prompt = (
                f"Generate a persona interested in news.\n"
                f"Name: {name_prompt}\n"
                f"Age: {ages[i]}\n"
                f"Gender: {gender_prompt}\n"
                f"Primary News Interest: {primary_interest}\n"
                f"Secondary News Interest: {secondary_interest}"
                f"Job: {jobs[i]}\n"
                f"Description: Provide a detailed description of their news consumption habits, secondary interests, and reasons for following news."
            )

            #name = self.generate_field(name_prompt)
            #gender = self.generate_field(gender_prompt)
            description = self.parse_description(self.generate_description(description_prompt))

            personas.append(description)

        return pd.DataFrame(personas)

    def generate_user_dataset(self, p, num, directory, file_name, seed=config.DEFAULT_SEED):
        self.seed(seed)
        num_iterations = num // 4
        file_path = os.path.join(directory, file_name)
        
        # Ensure the directory exists
        if not os.path.exists(directory):
            os.makedirs(directory)

        if os.path.exists(file_path):
            os.remove(file_path)

        for _ in range(num_iterations):
            df = self.generate_persona(seed, 4)
            df.to_csv(file_path, index=False, quoting=csv.QUOTE_ALL, mode="a", header=not os.path.exists(file_path))
            seed += 4

def parse_arguments():
    parser = argparse.ArgumentParser(description="Users generator for MIND dataset")
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    parser.add_argument("--num", type=int, default=config.DEFAULT_NUM)
    parser.add_argument("--split-without-news-preferences", type=float, default=config.DEFAULT_SPLIT)
    parser.add_argument("--file-name", type=str, default=config.DEFAULT_FILE_NAME)
    parser.add_argument("--dir", type=str, default=os.path.join("environment/mind", "users_generation/datasets"))
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    # Retrieve the API key from the environment variable
    api_key = os.getenv('LAS_API_TOKEN')
    if not api_key:
        raise ValueError("API key not found. Please set the 'LAS_API_TOKEN' environment variable.")

    generator = UserGenerator(api_key=api_key)
    generator.generate_user_dataset(
        args.split_without_news_preferences,
        args.num,
        args.dir,
        args.file_name,
        seed=args.seed,
    )
