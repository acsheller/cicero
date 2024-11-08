import os
from typing import List
from environment.LLM import LLMRater
from environment.LLM.llm import LLM
import re
from .third_person_descriptive_one_ten import ThirdPersonDescriptiveOneTen_OurSys

from environment.memory import UserNewsInteraction
from environment.mind import News, NewsLoader
from environment.users import User


class ThirdPersonDescriptiveOneTen_1Shot_OurSys(ThirdPersonDescriptiveOneTen_OurSys):
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

    def _get_few_shot_prompts(self):
        if self.cache_few_shot_prompts is None:
            base = os.path.dirname(os.path.abspath(__file__))
            items_loader = NewsLoader()

            user = User(
                "Alex",
                "M",
                25,
                (
                    "Alex is a voracious consumer of news, particularly in the realm"
                    " of law and justice. As a journalist, he feels a responsibility to"
                    " stay current on the latest developments in legal matters, court"
                    " decisions, and societal implications of justice-related issues."
                    " He subscribes to various law journals and frequently visits websites"
                    " like The New York Times, BBC News, and The Guardian for their coverage"
                    " of legal affairs.\n"

                    "In addition to his primary interest, Alex follows football news closely. Being"
                    " a lifelong fan of the sport, he enjoys keeping up with his favorite teams and"
                    " players, as well as the general state of the sport. He checks ESPN and other"
                    " sports news websites daily and often watches live matches during his free time.\n"

                    "Alex follows the news not only for his job but also due to a deep-rooted interest in"
                    " current affairs. He believes that staying informed about the world helps him to be a"
                    " responsible citizen and to understand the context of his own life better. He often engages"
                    " in discussions about news stories on social media, participates in local town hall meetings,"
                    " and occasionally writes opinion pieces for his local newspaper."                    
                ),
            )

            news_articles = items_loader.load_items_from_ids(['N55528', 'N16016', 'N61837', 'N53526'])
            news_article = news_articles[0]
            num_interacted = 0
            interactions = [
                UserNewsInteraction(5, 0, 1),
                UserNewsInteraction(8, 0, 1),
                UserNewsInteraction(5, 0, 1),
            ]
            retrieved_items = news_articles[1:]
            prompt = self._get_prompt(
                user,
                news_article,
                num_interacted,
                interactions,
                retrieved_items,
                do_rename=False,
            )
            explanation = (
                "Given Alex's reading habits and previous ratings, it is likely that"
                " he would rate the article 'The Brands Queen Elizabeth, Prince Charles, and" 
                " Prince Philip Swear By' a four on a scale of one to ten. Despite his primary interest in"
                " legal and justice-related news, Alex has demonstrated an appreciation for current affairs"
                " and lifestyle topics, albeit to a lesser extent. The article on the Royal Family's preferred"
                " brands offers an interesting glimpse into their personal preferences, which might capture" 
                " Alex’s curiosity to some degree. However, his focus on more impactful news, such as legal" 
                " matters and football, means that this lifestyle article would only moderately engage"
                " him, leading him to rate it a four."
            )
            self.cache_few_shot_prompts = [
                {"role": "user", "content": prompt[0]["content"]},
                {"role": "assistant", "content": prompt[1]["content"] + explanation},
            ]
        return self.cache_few_shot_prompts

class ThirdPersonDescriptiveOneTen_1Shot_OurSys_large(ThirdPersonDescriptiveOneTen_OurSys):
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

    def _get_few_shot_prompts(self):
        if self.cache_few_shot_prompts is None:
            base = os.path.dirname(os.path.abspath(__file__))
            items_loader = NewsLoader()

            user = User(
                "Alex",
                "M",
                25,
                (
                    "Alex is a voracious consumer of news, particularly in the realm"
                    " of law and justice. As a journalist, he feels a responsibility to"
                    " stay current on the latest developments in legal matters, court"
                    " decisions, and societal implications of justice-related issues."
                    " He subscribes to various law journals and frequently visits websites"
                    " like The New York Times, BBC News, and The Guardian for their coverage"
                    " of legal affairs.\n"

                    "In addition to his primary interest, Alex follows football news closely. Being"
                    " a lifelong fan of the sport, he enjoys keeping up with his favorite teams and"
                    " players, as well as the general state of the sport. He checks ESPN and other"
                    " sports news websites daily and often watches live matches during his free time.\n"

                    "Alex follows the news not only for his job but also due to a deep-rooted interest in"
                    " current affairs. He believes that staying informed about the world helps him to be a"
                    " responsible citizen and to understand the context of his own life better. He often engages"
                    " in discussions about news stories on social media, participates in local town hall meetings,"
                    " and occasionally writes opinion pieces for his local newspaper."                    
                ),
            )

            news_articles = items_loader.load_items_from_ids(['N37683', 'N8006', 'N54264', 'N53526'])
            news_article = news_articles[0]
            num_interacted = 0
            interactions = [
                UserNewsInteraction(5, 0, 1),
                UserNewsInteraction(8, 0, 1),
                UserNewsInteraction(5, 0, 1),
            ]
            retrieved_items = news_articles[1:]
            prompt = self._get_prompt(
                user,
                news_article,
                num_interacted,
                interactions,
                retrieved_items,
                do_rename=False,
            )
            explanation = (
                "Given Alex's reading habits and previous ratings, it is likely that"
                " he would rate the article 'The Brands Queen Elizabeth, Prince Charles, and" 
                " Prince Philip Swear By' a four on a scale of one to ten. Despite his primary interest in"
                " legal and justice-related news, Alex has demonstrated an appreciation for current affairs"
                " and lifestyle topics, albeit to a lesser extent. The article on the Royal Family's preferred"
                " brands offers an interesting glimpse into their personal preferences, which might capture" 
                " Alex’s curiosity to some degree. However, his focus on more impactful news, such as legal" 
                " matters and football, means that this lifestyle article would only moderately engage"
                " him, leading him to rate it a four."
            )
            self.cache_few_shot_prompts = [
                {"role": "user", "content": prompt[0]["content"]},
                {"role": "assistant", "content": prompt[1]["content"] + explanation},
            ]
        return self.cache_few_shot_prompts
