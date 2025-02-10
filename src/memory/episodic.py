from typing import List, Optional, Dict, Any
import numpy as np
from datetime import datetime
import heapq
from collections import defaultdict

from .types import EpisodicTrace, MemoryConfig, to_low_precision

class EpisodicMemory:
    """Manages episodic traces with salience-based storage and pruning."""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.traces: List[EpisodicTrace] = []
        self.trace_indices: Dict[str, int] = {}  # source_id -> index mapping
        self.cluster_traces: Dict[int, List[int]] = defaultdict(list)  # cluster_id -> trace indices
    
    def compute_salience(self, trace: EpisodicTrace) -> float:
        """Compute time-decayed salience score."""
        time_diff = (datetime.now() - trace.timestamp).total_seconds()
        time_factor = np.exp(-self.config.time_decay_factor * time_diff)
        return trace.salience * time_factor
    
    def add_trace(self, trace: EpisodicTrace) -> bool:
        """Add trace if it passes salience threshold."""
        # Ensure embedding is low precision
        trace.embedding = to_low_precision(trace.embedding, self.config.precision_dtype)
        
        # Check salience
        current_salience = self.compute_salience(trace)
        if current_salience < self.config.salience_threshold:
            return False
        
        # Add trace
        if trace.source_id and trace.source_id in self.trace_indices:
            # Update existing trace
            idx = self.trace_indices[trace.source_id]
            self.traces[idx] = trace
        else:
            # Add new trace
            self.traces.append(trace)
            if trace.source_id:
                self.trace_indices[trace.source_id] = len(self.traces) - 1
        
        # Update cluster mapping if cluster is assigned
        if trace.cluster_id is not None:
            self.cluster_traces[trace.cluster_id].append(len(self.traces) - 1)
        
        # Prune if needed
        if len(self.traces) > self.config.max_total_traces:
            self._prune_traces()
        
        return True
    
    def get_traces_for_cluster(self, cluster_id: int) -> List[EpisodicTrace]:
        """Get all traces for a given cluster."""
        indices = self.cluster_traces.get(cluster_id, [])
        return [self.traces[i] for i in indices if i < len(self.traces)]
    
    def update_cluster_assignment(self, trace_idx: int, cluster_id: int):
        """Update cluster assignment for a trace."""
        if 0 <= trace_idx < len(self.traces):
            old_cluster = self.traces[trace_idx].cluster_id
            if old_cluster is not None:
                self.cluster_traces[old_cluster].remove(trace_idx)
            
            self.traces[trace_idx].cluster_id = cluster_id
            self.cluster_traces[cluster_id].append(trace_idx)
    
    def get_batch_for_clustering(self, batch_size: Optional[int] = None) -> List[EpisodicTrace]:
        """Get batch of traces for clustering update."""
        batch_size = batch_size or self.config.batch_size
        return self.traces[-batch_size:]
    
    def _prune_traces(self):
        """Remove traces with lowest salience scores."""
        # Compute current salience for all traces
        salience_scores = [(self.compute_salience(trace), i) for i, trace in enumerate(self.traces)]
        
        # Keep only top traces
        keep_count = self.config.max_total_traces // 2  # Aggressive pruning
        top_indices = heapq.nlargest(keep_count, salience_scores, key=lambda x: x[0])
        
        # Update traces and mappings
        new_traces = []
        new_indices = {}
        new_cluster_traces = defaultdict(list)
        
        for _, old_idx in top_indices:
            trace = self.traces[old_idx]
            new_idx = len(new_traces)
            
            # Update trace list
            new_traces.append(trace)
            
            # Update source mapping
            if trace.source_id:
                new_indices[trace.source_id] = new_idx
            
            # Update cluster mapping
            if trace.cluster_id is not None:
                new_cluster_traces[trace.cluster_id].append(new_idx)
        
        # Replace old data structures
        self.traces = new_traces
        self.trace_indices = new_indices
        self.cluster_traces = new_cluster_traces
    
    def get_cluster_statistics(self, cluster_id: int) -> Dict[str, Any]:
        """Get statistics for a cluster."""
        traces = self.get_traces_for_cluster(cluster_id)
        if not traces:
            return {}
        
        salience_scores = [self.compute_salience(t) for t in traces]
        return {
            "size": len(traces),
            "avg_salience": np.mean(salience_scores),
            "max_salience": np.max(salience_scores),
            "min_salience": np.min(salience_scores),
            "recent_count": sum(1 for t in traces 
                              if (datetime.now() - t.timestamp).total_seconds() < 3600)
        } 