import typing

from environment.mind.news import News
from environment.memory import UserNewsInteraction
from abc import ABC, abstractmethod
import numpy as np
import json
from typing import List, Tuple

class ItemsRetrievalMind(ABC):
    """
    Object that is responsable to retrieve items
    """

    @abstractmethod
    def retrieve(
        self,
        curr_item: News,
        item_list: typing.List[News],
        interactions: typing.List[UserNewsInteraction],
    ) -> typing.Tuple[typing.List[News], typing.List[UserNewsInteraction]]:
        """
        The retrieve function is responsable to select from a list of items, with corresponding ratings, timestamp and num watched (i.e. interactions),
        a subset of items that are more relevant in order to construct a prompt for the LLM.
        """
        pass

class SimpleNewsRetrieval(ItemsRetrievalMind):
    """
    Object that is responsible for retrieving items, the items are picked based on a simple similarity

    Attributes:
        num (integer): maximum number of items to retrieve
    """

    def __init__(self, num: int):
        self.num = num

    # Method to calculate Jaccard similarity for lists
    def jaccard_similarity(self, list1, list2):
        set1, set2 = set(list1), set(list2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union != 0 else 1

    # Method to calculate similarity between two entities
    def compare_entities(self, entity1, entity2):
        # Handle missing fields by setting default values
        label1, label2 = entity1.get('Label', ''), entity2.get('Label', '')
        type1, type2 = entity1.get('Type', ''), entity2.get('Type', '')
        wikidata1, wikidata2 = entity1.get('WikidataId', ''), entity2.get('WikidataId', '')
        confidence1, confidence2 = entity1.get('Confidence', 0), entity2.get('Confidence', 0)
        occurrence1, occurrence2 = entity1.get('OccurrenceOffsets', []), entity2.get('OccurrenceOffsets', [])
        surface_forms1, surface_forms2 = entity1.get('SurfaceForms', []), entity2.get('SurfaceForms', [])

        label_similarity = 1 if label1 == label2 else 0
        type_similarity = 1 if type1 == type2 else 0
        wikidata_similarity = 1 if wikidata1 == wikidata2 else 0
        confidence_similarity = 1 - abs(confidence1 - confidence2)
        occurrence_similarity = self.jaccard_similarity(occurrence1, occurrence2)
        surface_forms_similarity = self.jaccard_similarity(surface_forms1, surface_forms2)
        
        return np.mean([
            label_similarity,
            type_similarity,
            wikidata_similarity,
            confidence_similarity,
            occurrence_similarity,
            surface_forms_similarity
        ])

    # Method to calculate similarity between two News items
    def similarity(self, item1: News, item2: News):
        """
        Similarity function used to retrieve items, it constructs a greedy similarity based on
        the entities, category, subcategory, vote average, and director

        Args:
            item1 (News): first item
            item2 (News): second item

        Return:
            similarity score (float)
        """
        # Compare title entities
        if len(item1.title_entities) > 0 and len(item2.title_entities) > 0:
            ent_list1 = json.loads(item1.title_entities)
            ent_list2 = json.loads(item2.title_entities)
            if ent_list1 and ent_list2:
                entities_similarity = self.compare_entities(ent_list1[0], ent_list2[0])
            else:
                entities_similarity = 0


        # Compare categories
        category_similarity = 1 if item1.category == item2.category else 0

        # Compare subcategories
        subcategory_similarity = 1 if item1.subcategory == item2.subcategory else 0


        return np.mean([
            entities_similarity,
            category_similarity,
            subcategory_similarity
        ])

    # Method to retrieve the most relevant items
    def retrieve(
        self,
        curr_item: News,
        item_list: List[News],
        interactions: List[UserNewsInteraction],
    ) -> Tuple[List[News], List[UserNewsInteraction]]:
        """
        The retrieve function is responsible for retrieving the most relevant items, based on the similarity function.
        It sorts the items in decreasing order based on the similarity with curr_item and picks the most relevant (num of them).

        Args:
            curr_item (News): the item of interest
            item_list (List[News]): a list of News that the user has seen, from which we want to select the most similar to curr_item
            interactions (List[UserNewsInteraction]): a list containing all the interactions of the user, the order in the list should correspond with
                the order of item_list

        Return:
            retrieved_items (List[News]): list containing the most relevant items
            retrieved_interactions (List[UserNewsInteraction]): list containing interactions corresponding to the items in retrieved_items
        """
        tmp_list = []
        for item, interaction in zip(item_list, interactions):
            tmp_list.append((self.similarity(item, curr_item), item, interaction))

        tmp_list.sort(key=lambda x: x[0], reverse=True)

        retrieved_items = []
        retrieved_interactions = []

        for i, (similarity, item, interaction) in enumerate(tmp_list):
            if i >= self.num:
                break
            retrieved_items.append(item)
            retrieved_interactions.append(interaction)

        return retrieved_items, retrieved_interactions

class TimeItemsRetrievalMind(ItemsRetrievalMind):
    """
    Object responsable to retrieve items, the items are retrieved are always the most recent ones.

    Attributes:
        num (integer): maximum number of items to retrieve
    """

    def __init__(self, num: int):
        self.num = num

    def retrieve(
        self,
        curr_item: News,
        item_list: typing.List[News],
        interactions: typing.List[UserNewsInteraction],
    ) -> typing.Tuple[typing.List[News], typing.List[UserNewsInteraction]]:
        '''
        The retrieve function is responsable to retrieve most relevant items, in this case it is based on the time.
        It sort the items in decreasing order based on the time  and picks the most relevant (num of them)
        Args:
            curr_item (Movie): the item of interest
            item_list (List Movie): a list of Movies that the user has seen, from which we want to select the most recent ones
            interactions (List Interaction): a list containing all the interaction of the user, the order in the list should correspond with th
                the order of item_list
        Return:
            retrieved_items (List Movie): list containing the most recent items
            retrieved_interactions (List Interaction): lis containing interactions corresponding to the items in retrieved_items
        """
        '''
        tmp_list = list(zip(item_list, interactions))
        tmp_list.sort(key=lambda x: x[1].timestamp, reverse=True)

        retrieved_items = []
        retrieved_interactions = []

        for i, (item, interaction) in enumerate(tmp_list):
            if i >= self.num:
                break
            retrieved_items.append(item)
            retrieved_interactions.append(interaction)

        return retrieved_items, retrieved_interactions

class BestWorstItemsRetrievalMind(ItemsRetrievalMind):
    """
    This is a special retrieval since it only returns the two films that re considere the best and the worst from the user.
    """

    def __init__(self):
        pass

    def retrieve(
        self,
        curr_item: News,
        item_list: typing.List[News],
        interactions: typing.List[UserNewsInteraction],
    ) -> typing.Tuple[typing.List[News], typing.List[UserNewsInteraction]]:
        """
        Function that returns the best item with higher score, and the item with lowest score (only the seen items). In this case
        the only two items retrieved are the ones considered the worst and the best from the user.

        Args:
            item_list (list News): list of item seen
            scores (list of integer): list of the scores of the items in item_list

        Return
            List of News containing:
                items_scores[max_idx][0] (News): item with the highest score
                items_scores[min_idx][0] (Movie): item with the lowest score
            List of Interactions containing:
                items_scores[max_idx][1] (integer): interaction (score, time) of best item
                items_scores[min_idx][1] (integer): interaction (score, time) of worst item
        """
        items_interactions = [
            (m, i) for (m, i) in zip(item_list, interactions) if i.score > 0
        ]
        if len(items_interactions) == 0:
            return None, None, None, None

        max_idx = 0
        min_idx = 0
        for i, (m, s) in enumerate(items_interactions):
            max_idx = i if s > items_interactions[max_idx][1].score else max_idx
            min_idx = i if s < items_interactions[min_idx][1].score else min_idx

        return (
            [items_interactions[max_idx][0], items_interactions[min_idx][0]],
            [items_interactions[max_idx][1], items_interactions[min_idx][1]],
        )

class SentenceSimilarityItemsRetrievalMind(ItemsRetrievalMind):
    def __init__(self, num: int, name_field_embedding: str) -> None:
        self.num = num
        self.name_field_embedding = name_field_embedding
        super().__init__()

    def retrieve(
        self,
        curr_item: News,
        item_list: typing.List[News],
        interactions: typing.List[UserNewsInteraction],
    ) -> typing.Tuple[typing.List[News], typing.List[UserNewsInteraction]]:
        """
        The retrieve function is responsable to select from a list of items, with corresponding ratings, timestamp and num watched (i.e. interactions),
        a subset of items that are more relevant in order to construct a prompt for the LLM.
        """

        def f(x: News):
            # cosine similarity
            c = np.array(curr_item.__getattribute__(self.name_field_embedding))
            x = np.array(x.__getattribute__(self.name_field_embedding))
            return np.dot(c, x) / (np.linalg.norm(c) * np.linalg.norm(x))

        item_interactions = list(zip(item_list, interactions))
        item_interactions.sort(key=lambda x: f(x[0]), reverse=True)

        return (
            [item for item, _ in item_interactions[: self.num]],
            [interaction for _, interaction in item_interactions[: self.num]],
        )
