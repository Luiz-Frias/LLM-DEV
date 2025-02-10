from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any
import numpy as np
import torch
from datetime import datetime

def to_low_precision(tensor):
    """Convert tensor to float16 for memory efficiency."""
    if isinstance(tensor, torch.Tensor):
        return tensor.to(torch.float16)
    elif isinstance(tensor, np.ndarray):
        return tensor.astype(np.float16)
    else:
        raise TypeError(f"Unsupported type for to_low_precision: {type(tensor)}")

@dataclass
class EpisodicTrace:
    """Single memory trace with metadata."""
    
    # Core content
    embedding: np.ndarray  # Low precision embedding
    text: str  # Original text or summary
    
    # Metadata
    timestamp: datetime
    salience: float  # Importance score [0, 1]
    cluster_id: Optional[int] = None  # Assigned cluster
    source_id: Optional[str] = None  # Original source/context
    
    # Optional metadata
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Ensure embedding is in low precision."""
        if isinstance(self.embedding, torch.Tensor):
            self.embedding = self.embedding.detach().cpu().numpy()
        if not isinstance(self.embedding, np.ndarray):
            raise TypeError(f"Embedding must be a numpy array, got {type(self.embedding)}")
        if self.embedding.dtype != np.float16:
            self.embedding = self.embedding.astype(np.float16)

@dataclass
class ClusterSummary:
    """Summary of a cluster of episodic traces."""
    
    cluster_id: int
    centroid: np.ndarray  # Low precision centroid
    size: int  # Number of traces in cluster
    
    # Cluster statistics
    avg_salience: float
    creation_time: datetime
    last_updated: datetime
    
    # Semantic information
    keywords: List[str]
    summary_text: Optional[str] = None
    
    def __post_init__(self):
        """Ensure centroid is in low precision."""
        if not isinstance(self.centroid, np.ndarray):
            self.centroid = to_low_precision(self.centroid)
        elif self.centroid.dtype != np.float16:
            self.centroid = to_low_precision(self.centroid)

@dataclass
class MemoryConfig:
    """Configuration for memory components."""
    
    # Embedding settings
    embedding_dim: int = 32  # Reduced from typical 768
    precision_dtype: Any = np.float16
    
    # Salience settings
    salience_threshold: float = 0.3  # Lowered from 0.6 to allow more traces
    time_decay_factor: float = 0.1
    
    # Clustering settings
    n_clusters: int = 10
    batch_size: int = 16
    min_cluster_size: int = 3
    similarity_threshold: float = 0.5
    
    # Pruning settings
    max_traces_per_cluster: int = 100
    max_total_traces: int = 1000
    prune_threshold: float = 0.3
    
    # Knowledge graph settings
    min_edge_weight: float = 0.4
    max_edges_per_node: int = 5 