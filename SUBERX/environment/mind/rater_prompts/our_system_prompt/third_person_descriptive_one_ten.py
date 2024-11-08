import os
from typing import List
from environment.LLM import LLMRater
from environment.LLM.llm import LLM
import re
import pandas as pd
from environment.memory import UserNewsInteraction
from environment.mind import News, NewsLoader
from environment.users import User


class ThirdPersonDescriptiveOneTen_OurSys(LLMRater):
    def __init__(
        self,
        llm: LLM,
        current_items_features_list,
        previous_items_features_list,
        llm_render=False,
        llm_query_explanation=False,
    ):
        super().__init__(
            llm,
            current_items_features_list,
            previous_items_features_list,
            llm_render,
            llm_query_explanation,
        )
        self.cache_few_shot_prompts = None
        self.request_scale = "one-ten"

        self.system_prompt = (
            "You are a highly sophisticated news rating assistant, equipped with an"
            " advanced understanding of human behavior. Your mission is to deliver"
            " personalized news recommendations by carefully considering the unique"
            " characteristics, tastes, and previously seen news of each individual. When"
            " presented with information about a specific news article, you will diligently"
            " analyze its primary categories, persons, places, and average rating. Using this"
            " comprehensive understanding, your role is to provide thoughtful and"
            " accurate ratings for news on a scale of one to ten, ensuring they resonate"
            " with the person's preferences. Remain"
            " impartial and refrain from introducing any biases in your predictions."
            " You are an impartial and reliable source of news rating predictions for"
            " the given individual."
        )

    def adjust_rating_in(self, rating):
        return rating -1

    def adjust_rating_out(self, rating):
        return rating +1

    def adjust_text_in(self, text, do_rename=True):
        text = re.sub("\d+", lambda x: f"{int(x.group())-1}", text)
        if do_rename:
            text = text.replace("Alex", "Michael")
            text = text.replace("Nicholas", "Michael")
        return text

    def _get_prompt(
        self,
        user: User,
        news: News,
        num_interacted: int,
        interactions: List[UserNewsInteraction],
        retrieved_items: List[News],
        do_rename=True,
    ):
        if user.gender == "M":
            gender = "man"
            pronoun = "he"
            if int(user.age) < 18:
                gender = "boy"
        else:
            gender = "woman"
            pronoun = "she"
            if int(user.age) < 18:
                gender = "girl"

        item_interaction = ""  # NOTE it should be parametrized
        for m, i in zip(retrieved_items, interactions):
            item_interaction += (
                f'"{m.title}" ({int(self.adjust_rating_in(i.rating))}), '
            )
        if len(retrieved_items) > 0:
            item_interaction = item_interaction[:-2]  # remove last comma


        if 'small' in NewsLoader._dataset:
            news_cats_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "../../datasets/news_cats_small.csv",)   
        else:
            news_cats_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "../../datasets/news_cats.csv",)

        news_catagories_list = pd.read_csv(news_cats_path, header=None)[0].tolist()
        news_catagories_list_string = ""
        for g in news_catagories_list:
            
            news_catagories_list_string += f"-{g}\n"




  
        if len(news.abstract) > 0:
            overview = news.abstract[0].lower() + news.abstract[1:]
        else:
            overview = ""

        name = user.name.split(" ")[0]
        # NOTE: this is a hack to make sure that the name is not the same as the 2 possible names used in the few-shot prompts
        name = self.adjust_text_in(name, do_rename)

        prompt = (
            f"{name} is a {user.age} years old {gender},"
            f" {pronoun} is {self.adjust_text_in(user.description, do_rename)}\n"
            + (
                f"{name} has previously read the following news articles (in"
                " parentheses are the ratings he gave on a scale of one to ten):"
                f" {item_interaction}.\n"
                if len(retrieved_items) > 0
                and len(self.previous_items_features_list) > 0
                else ""
            )
            + f'\nConsider the news article entitled "{news.title}".'
              f" It is described as follows: {overview}\n,"
            + (
                f' The news article is catagorized  "{news.category}" with a subcatagory of  "{news.subcategory}:'
                f' On average the news article has a click-through rate of "{news.click_through_rate}" and'
                f" the news article has been read {news.read_frequency} times.\n\n"
              
            )
            + f' {name} has read the news article "{news.title}" for the'
            f" {self.number_to_rank(num_interacted+1)} times.\n"
            + f"What can you conclude about {name}'s rating for the news article"
            f' "{news.title}" on a scale of one to ten, where one represents a low'
            " rating and ten represents a high rating, based on available information"
            " and logical reasoning?"
        )

        initial_assistant = (
            f"Based on {name}'s preferences and tastes, I conclude that {pronoun} will"
            " assign a rating of "
        )

        return [
            {"role": "user", "content": prompt},
            {"role": "assistant_start", "content": initial_assistant},
        ]

    def _get_few_shot_prompts(self):
        return []

    def _get_prompt_explanation(self, prompt, rating):
        # map 1 to 10 from number to text
        m = {
            1: "one",
            2: "two",
            3: "three",
            4: "four",
            5: "five",
            6: "six",
            7: "seven",
            8: "eight",
            9: "nine",
            10: "ten",
        }

        initial_explanation = f"{m[rating]} on a scale of one to ten, because "
        prompt[1]["content"] += initial_explanation
        return prompt
