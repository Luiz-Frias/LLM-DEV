import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict
import networkx as nx
import numpy as np
from .types import EpisodicTrace, ClusterSummary
from .hierarchical_memory import HierarchicalMemoryTree

class SymbolicSummarizer:
    """
    Generates symbolic summaries from memory components using a hybrid approach
    combining neural embeddings with symbolic reasoning.
    """
    def __init__(
        self,
        max_summary_tokens: int = 8,
        min_cluster_size: int = 3,
        min_edge_weight: float = 0.4,
        device: Optional[torch.device] = None
    ):
        self.max_summary_tokens = max_summary_tokens
        self.min_cluster_size = min_cluster_size
        self.min_edge_weight = min_edge_weight
        self.device = device or torch.device("cpu")
        
        # Cache for computed summaries
        self.summary_cache: Dict[int, str] = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def extract_key_concepts(
        self,
        traces: List[EpisodicTrace],
        cluster_summaries: List[ClusterSummary]
    ) -> List[str]:
        """Extract key concepts from traces and cluster summaries."""
        # Collect all text content
        all_text = " ".join([
            trace.text for trace in traces
        ] + [
            summary.summary_text for summary in cluster_summaries
            if summary.summary_text is not None
        ])
        
        # Extract keywords using simple frequency analysis
        # This could be enhanced with more sophisticated NLP
        words = all_text.lower().split()
        word_freq = defaultdict(int)
        for word in words:
            if len(word) > 3:  # Skip short words
                word_freq[word] += 1
        
        # Sort by frequency and return top keywords
        sorted_words = sorted(
            word_freq.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [word for word, _ in sorted_words[:10]]  # Top 10 keywords
    
    def summarize_cluster(
        self,
        cluster_id: int,
        traces: List[EpisodicTrace],
        centroid: torch.Tensor
    ) -> str:
        """Generate a symbolic summary for a cluster of traces."""
        # Check cache first
        if cluster_id in self.summary_cache:
            self.cache_hits += 1
            return self.summary_cache[cluster_id]
        
        self.cache_misses += 1
        
        # Extract common themes from trace texts
        all_text = " ".join([trace.text for trace in traces])
        words = all_text.lower().split()
        word_freq = defaultdict(int)
        for word in words:
            if len(word) > 3:
                word_freq[word] += 1
        
        # Get top keywords
        top_keywords = sorted(
            word_freq.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]  # Top 5 keywords
        
        # Create summary
        summary = f"Cluster {cluster_id}: " + ", ".join([
            f"{word} ({freq})" for word, freq in top_keywords
        ])
        
        # Cache the summary
        self.summary_cache[cluster_id] = summary
        return summary
    
    def summarize_memory_tree(
        self,
        memory_tree: HierarchicalMemoryTree,
        traces: List[EpisodicTrace]
    ) -> List[str]:
        """Generate summaries for memory tree nodes."""
        summaries = []
        
        def _summarize_node(node, depth=0):
            if node is None:
                return
            
            # Only summarize nodes with sufficient traces
            if node.count >= self.min_cluster_size:
                # Find traces closest to this node's centroid
                node_traces = []
                for trace in traces:
                    if isinstance(trace.embedding, np.ndarray):
                        trace_tensor = torch.from_numpy(trace.embedding)
                    else:
                        trace_tensor = trace.embedding
                    
                    # Compare with node centroid
                    similarity = torch.nn.functional.cosine_similarity(
                        trace_tensor.view(1, -1),
                        node.centroid.view(1, -1)
                    ).item()
                    
                    if similarity > 0.7:  # Threshold for trace assignment
                        node_traces.append(trace)
                
                if node_traces:
                    summary = self.summarize_cluster(
                        len(summaries),
                        node_traces,
                        node.centroid
                    )
                    summaries.append(summary)
            
            # Recursively summarize children
            _summarize_node(node.left, depth + 1)
            _summarize_node(node.right, depth + 1)
        
        _summarize_node(memory_tree.root)
        return summaries
    
    def summarize_knowledge_graph(
        self,
        knowledge_graph: nx.Graph,
        traces: List[EpisodicTrace]
    ) -> List[str]:
        """Generate summaries from knowledge graph structure."""
        summaries = []
        
        # Find important nodes using PageRank
        pagerank = nx.pagerank(knowledge_graph)
        important_nodes = sorted(
            pagerank.items(),
            key=lambda x: x[1],
            reverse=True
        )[:self.max_summary_tokens]
        
        # Generate summaries for important nodes
        for node_id, importance in important_nodes:
            node_data = knowledge_graph.nodes[node_id]
            if 'traces' in node_data:
                node_traces = node_data['traces']
                if len(node_traces) >= self.min_cluster_size:
                    summary = self.summarize_cluster(
                        node_id,
                        node_traces,
                        torch.tensor(node_data['centroid'])
                    )
                    summaries.append(summary)
        
        return summaries
    
    def get_symbolic_tokens(
        self,
        memory_tree: HierarchicalMemoryTree,
        knowledge_graph: nx.Graph,
        traces: List[EpisodicTrace],
        cluster_summaries: List[ClusterSummary]
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Get symbolic tokens and their importance weights
        from memory components.
        """
        # Get summaries from different components
        tree_summaries = self.summarize_memory_tree(memory_tree, traces)
        graph_summaries = self.summarize_knowledge_graph(knowledge_graph, traces)
        key_concepts = self.extract_key_concepts(traces, cluster_summaries)
        
        # Combine all symbolic information
        all_symbols = tree_summaries + graph_summaries + key_concepts
        
        # Calculate importance weights
        weights = {}
        for symbol in all_symbols:
            # Simple weighting based on length and frequency
            weight = min(1.0, len(symbol.split()) / 10)  # Longer summaries get more weight
            weights[symbol] = weight
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        return all_symbols, weights
    
    def get_state(self) -> Dict:
        """Get serializable state of the summarizer."""
        return {
            'max_summary_tokens': self.max_summary_tokens,
            'min_cluster_size': self.min_cluster_size,
            'min_edge_weight': self.min_edge_weight,
            'summary_cache': self.summary_cache,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses
        }
    
    def set_state(self, state: Dict):
        """Set the state of the summarizer from a dictionary."""
        self.max_summary_tokens = state['max_summary_tokens']
        self.min_cluster_size = state['min_cluster_size']
        self.min_edge_weight = state['min_edge_weight']
        self.summary_cache = state['summary_cache']
        self.cache_hits = state['cache_hits']
        self.cache_misses = state['cache_misses'] 