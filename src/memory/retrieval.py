import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
import numpy as np
from .hierarchical_memory import HierarchicalMemoryTree, cosine_similarity
from .types import EpisodicTrace, to_low_precision

class MemoryRetrieval:
    """
    Memory retrieval system that combines hierarchical memory tree
    and knowledge graph to efficiently fetch relevant context.
    """
    def __init__(
        self,
        max_retrieved_items: int = 8,
        similarity_threshold: float = 0.6,
        salience_threshold: float = 0.5,
        device: Optional[torch.device] = None
    ):
        self.max_retrieved_items = max_retrieved_items
        self.similarity_threshold = similarity_threshold
        self.salience_threshold = salience_threshold
        self.device = device or torch.device("cpu")
    
    def retrieve_from_tree(
        self,
        query_embedding: torch.Tensor,
        memory_tree: HierarchicalMemoryTree,
        top_k: int = 4
    ) -> List[torch.Tensor]:
        """
        Retrieve most similar items from hierarchical memory tree.
        Uses tree structure for efficient search.
        """
        def _search_node(node, query_embedding, results, min_similarity):
            if node is None:
                return
            
            # Compute similarity with current node
            similarity = cosine_similarity(query_embedding, node.centroid)
            
            if similarity >= min_similarity:
                results.append((similarity, node.centroid))
                
                # Search children if they might have better matches
                if node.left is not None:
                    _search_node(node.left, query_embedding, results, min_similarity)
                if node.right is not None:
                    _search_node(node.right, query_embedding, results, min_similarity)
        
        results = []
        _search_node(memory_tree.root, query_embedding, results, self.similarity_threshold)
        
        # Sort by similarity and return top-k centroids
        results.sort(key=lambda x: x[0], reverse=True)
        return [centroid for _, centroid in results[:top_k]]
    
    def retrieve_from_knowledge_graph(
        self,
        query_embedding: torch.Tensor,
        knowledge_graph,
        top_k: int = 4
    ) -> List[torch.Tensor]:
        """
        Retrieve relevant nodes from knowledge graph based on query.
        """
        similarities = []
        
        # Get all node embeddings from knowledge graph
        for node_id, data in knowledge_graph.graph.nodes(data=True):
            if 'centroid' in data:
                centroid = torch.tensor(data['centroid'], device=self.device)
                similarity = cosine_similarity(query_embedding, centroid)
                similarities.append((similarity, centroid))
        
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [centroid for _, centroid in similarities[:top_k]]
    
    def get_global_memory_tokens(
        self,
        query_embedding: torch.Tensor,
        memory_tree: HierarchicalMemoryTree,
        knowledge_graph,
        traces: List[EpisodicTrace]
    ) -> torch.Tensor:
        """
        Combine retrievals from different memory components into global tokens.
        """
        # Get retrievals from different components
        tree_retrievals = self.retrieve_from_tree(query_embedding, memory_tree)
        graph_retrievals = self.retrieve_from_knowledge_graph(query_embedding, knowledge_graph)
        
        # Get high salience recent traces
        recent_traces = []
        for trace in reversed(traces[-10:]):  # Look at last 10 traces
            if trace.salience >= self.salience_threshold:
                if isinstance(trace.embedding, np.ndarray):
                    trace_tensor = torch.from_numpy(trace.embedding)
                else:
                    trace_tensor = trace.embedding
                recent_traces.append(trace_tensor)
                if len(recent_traces) >= 4:  # Limit to 4 recent traces
                    break
        
        # Combine all retrievals
        all_retrievals = tree_retrievals + graph_retrievals + recent_traces
        
        # If we have more than max_retrieved_items, prioritize based on similarity
        if len(all_retrievals) > self.max_retrieved_items:
            similarities = [
                cosine_similarity(query_embedding, retrieval)
                for retrieval in all_retrievals
            ]
            # Get indices of top-k similarities
            top_k_indices = np.argsort(similarities)[-self.max_retrieved_items:]
            all_retrievals = [all_retrievals[i] for i in top_k_indices]
        
        # Stack retrievals into a single tensor
        if all_retrievals:
            memory_tokens = torch.stack(all_retrievals)
            # Add batch dimension if needed
            if memory_tokens.dim() == 2:
                memory_tokens = memory_tokens.unsqueeze(0)
            return to_low_precision(memory_tokens)
        else:
            # Return empty tensor if no retrievals
            return torch.empty(
                (1, 0, query_embedding.size(-1)),
                device=self.device,
                dtype=torch.float16
            )
    
    def get_state(self) -> Dict:
        """Get serializable state of the retrieval system."""
        return {
            'max_retrieved_items': self.max_retrieved_items,
            'similarity_threshold': self.similarity_threshold,
            'salience_threshold': self.salience_threshold
        }
    
    def set_state(self, state: Dict):
        """Set the state of the retrieval system from a dictionary."""
        self.max_retrieved_items = state['max_retrieved_items']
        self.similarity_threshold = state['similarity_threshold']
        self.salience_threshold = state['salience_threshold'] 