from abc import ABC
from .graph_extraction import GraphRAGExtractor, parse_fn, KG_TRIPLET_EXTRACT_TMPL
from .graph_store import GraphRAGStore
from llama_index.core import PropertyGraphIndex
from llama_index.core.schema import TextNode
from typing import List, Optional, Any

class GraphRAGBuilder(ABC):
    
    """
    Build a graph from text data with the help of an LLM and a graph store Neo4jPropertyGraphStore.
    """

    def __init__(self, llm: Any, extract_prompt: str = KG_TRIPLET_EXTRACT_TMPL, parse_fn: Any = parse_fn, 
                 max_paths_per_chunk: int = 10, num_workers: Optional[int] = None, 
                 username: str = "neo4j", password: str = "password", url: str = "neo4j://127.0.0.1:7687",
                 embed_model: Optional[Any] = None):
        super().__init__()

        self.llm = llm
        self.extract_prompt = extract_prompt
        self.parse_fn = parse_fn
        self.max_paths_per_chunk = max_paths_per_chunk
        self.num_workers = num_workers
        
        # Create the graph extractor
        self.kg_extractor = GraphRAGExtractor(
            llm=self.llm,
            extract_prompt=self.extract_prompt,
            max_paths_per_chunk=self.max_paths_per_chunk,
            parse_fn=self.parse_fn,
        )
        
        # Create the graph store
        self.graph_store = GraphRAGStore(
            self.llm,
            username=username, 
            password=password, 
            url=url
        )
        # Store the embedding model
        self.embed_model = embed_model
        
        # The property graph index
        self.index = None

        self.restore_index()
    
    def restore_index(self):

        self.index = PropertyGraphIndex.from_existing(
            llm=self.llm,
            kg_extractors=[self.kg_extractor],
            property_graph_store=self.graph_store,
            embed_model=self.embed_model,
        )
        self.graph_store.build_communities()

    def build_graph(self, nodes: List[TextNode], show_progress: bool = True) -> PropertyGraphIndex:
        """
        Build a graph from a list of TextNode objects.
        
        Args:
            nodes: List of TextNode objects to build the graph from
            show_progress: Whether to show a progress bar
            
        Returns:
            The PropertyGraphIndex object
        """
        self.index = PropertyGraphIndex(
            nodes=nodes,
            llm=self.llm,
            kg_extractors=[self.kg_extractor],
            property_graph_store=self.graph_store,
            embed_model=self.embed_model,
        )
        return self.index

    def add_nodes(self, new_nodes: List[TextNode], show_progress: bool = True) -> PropertyGraphIndex:
        """
        Add new nodes to an existing graph.
        
        Args:
            new_nodes: List of new TextNode objects to add to the graph
            show_progress: Whether to show a progress bar
            
        Returns:
            The updated PropertyGraphIndex
        """
        
        return self.build_graph(new_nodes, show_progress)
        
    
    def refresh_index(self):
        """
        Load the index again.
        """
        self.restore_index()