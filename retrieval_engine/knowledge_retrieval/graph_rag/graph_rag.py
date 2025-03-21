import re
from typing import List
from llama_index.core.llms import LLM
from llama_index.core import PropertyGraphIndex
from .graph_store import GraphRAGStore
from retrieval_engine.knowledge_retrieval.base_rag import BaseRAG

class GraphRAGQueryEngine(BaseRAG):

    def __init__(self, model_name = None, 
                    graph_store: GraphRAGStore = None, 
                    index: PropertyGraphIndex = None, 
                    llm: LLM = None, 
                    similarity_top_k: int = 20):
        
        super().__init__(model_name)
        self.graph_store = graph_store
        self.index = index
        self.llm = llm
        self.similarity_top_k = similarity_top_k

    def retrieve(self, query_str: str) -> List[str]:
        """Process all community summaries to generate answers to a specific query."""
        entities = self.get_entities(query_str, self.similarity_top_k)
       
        community_ids = self.retrieve_entity_communities(
            self.graph_store.entity_info, entities
        )
        community_summaries = self.graph_store.get_community_summaries()
        community_answers = [
            community_summary
            for id, community_summary in community_summaries.items()
            if id in community_ids
        ]
        return community_answers
    
    def rerank(self, query_str: str, top_k: int = 3) -> List[str]:
        """Rerank the retrieved entities based on the query."""
        ...

    def process(self, query_str: str, top_k: int = 3) -> List[str]:
        """Process the query and return the reranked results."""
        return self.retrieve(query_str)

    def get_entities(self, query_str, similarity_top_k):
        nodes_retrieved = self.index.as_retriever(
            similarity_top_k=similarity_top_k
        ).retrieve(query_str)

        enitites = set()
        pattern = (
            r"^(\w+(?:\s+\w+)*)\s*->\s*([a-zA-Z\s]+?)\s*->\s*(\w+(?:\s+\w+)*)$"
        )

        for node in nodes_retrieved:
            matches = re.findall(
                pattern, node.text, re.MULTILINE | re.IGNORECASE
            )
            for match in matches:
                subject = match[0]
                obj = match[2]
                enitites.add(subject)
                enitites.add(obj)

        return list(enitites)

    def retrieve_entity_communities(self, entity_info, entities):
        """
        Retrieve cluster information for given entities, allowing for multiple clusters per entity.

        Args:
        entity_info (dict): Dictionary mapping entities to their cluster IDs (list).
        entities (list): List of entity names to retrieve information for.

        Returns:
        List of community or cluster IDs to which an entity belongs.
        """
        community_ids = []

        for entity in entities:
            if entity in entity_info:
                community_ids.extend(entity_info[entity])
        return list(set(community_ids))