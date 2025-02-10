from typing import List, Dict, Optional, Tuple
import numpy as np
from datetime import datetime
from sklearn.cluster import MiniBatchKMeans
from collections import defaultdict
from dataclasses import asdict
import torch

from .types import EpisodicTrace, ClusterSummary, MemoryConfig, to_low_precision

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # Only convert to float32 for the final computation
    a = a.astype(np.float16)
    b = b.astype(np.float16)
    # Use float32 only for the final dot product and norm to avoid numerical instability
    return float(np.dot(a.astype(np.float32), b.astype(np.float32)) / 
                (np.linalg.norm(a.astype(np.float32)) * np.linalg.norm(b.astype(np.float32)) + 1e-8))

class DynamicClusterer:
    """Manages incremental clustering of episodic traces."""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.kmeans = MiniBatchKMeans(
            n_clusters=config.n_clusters,
            batch_size=config.batch_size,
            compute_labels=True,
            random_state=42
        )
        self._initialized = False
        self.summaries: Dict[int, ClusterSummary] = {}
        self.trace_counts: Dict[int, int] = defaultdict(int)
        self.total_salience: Dict[int, float] = defaultdict(float)
        
    def get_state(self) -> Dict:
        """Get serializable state of the clusterer."""
        state = {
            'initialized': self._initialized,
            'summaries': {k: asdict(v) for k, v in self.summaries.items()},
            'trace_counts': dict(self.trace_counts),
            'total_salience': dict(self.total_salience)
        }
        
        # Only include kmeans state if initialized
        if self._initialized and hasattr(self.kmeans, 'cluster_centers_'):
            state.update({
                'kmeans_cluster_centers': self.kmeans.cluster_centers_.tolist(),
                'kmeans_counts': self.kmeans.counts_.tolist() if hasattr(self.kmeans, 'counts_') else None
            })
        
        return state
    
    def set_state(self, state: Dict):
        """Set the state of the clusterer from a dictionary."""
        self._initialized = state['initialized']
        
        # Restore kmeans state if it was initialized
        if self._initialized and 'kmeans_cluster_centers' in state:
            self.kmeans.cluster_centers_ = np.array(state['kmeans_cluster_centers'], dtype=np.float32)
            if state['kmeans_counts'] is not None:
                self.kmeans.counts_ = np.array(state['kmeans_counts'])
            
        self.summaries = {
            k: ClusterSummary(
                cluster_id=v['cluster_id'],
                centroid=np.array(v['centroid'], dtype=np.float16),
                size=v['size'],
                avg_salience=v['avg_salience'],
                creation_time=datetime.fromisoformat(v['creation_time']),
                last_updated=datetime.fromisoformat(v['last_updated']),
                keywords=v['keywords'],
                summary_text=v['summary_text']
            ) for k, v in state['summaries'].items()
        }
        self.trace_counts = defaultdict(int, state['trace_counts'])
        self.total_salience = defaultdict(float, state['total_salience'])
        
    def partial_fit(self, input_data):
        """Incrementally update clusters with new data."""
        if isinstance(input_data, list) and isinstance(input_data[0], EpisodicTrace):
            # Handle list of traces
            if not input_data:
                return []
            embeddings = np.stack([t.embedding for t in input_data])
        elif isinstance(input_data, torch.Tensor):
            # Handle tensor input
            if len(input_data) == 0:
                return []
            embeddings = input_data.detach().cpu().numpy()
        else:
            embeddings = np.array(input_data)
        
        # Store in float16 by default
        embeddings = embeddings.astype(np.float16)
        
        # Convert to float32 only for kmeans operation
        embeddings_f32 = embeddings.astype(np.float32)
        
        # Update clusters
        if not self._initialized:
            self.kmeans.partial_fit(embeddings_f32)
            self._initialized = True
        else:
            self.kmeans.partial_fit(embeddings_f32)
        
        # Get cluster assignments
        labels = self.kmeans.labels_
        
        # Update statistics if we have traces
        if isinstance(input_data, list) and isinstance(input_data[0], EpisodicTrace):
            for trace, label in zip(input_data, labels):
                self._update_cluster_stats(trace, label)
        
        return labels.tolist()
    
    def predict(self, input_data):
        """
        Predict cluster for new data points
        Args:
            input_data: Either a tensor, numpy array, or an EpisodicTrace object
        Returns:
            Predicted cluster ID
        """
        if isinstance(input_data, torch.Tensor):
            embedding = input_data.detach().cpu().numpy()
        elif isinstance(input_data, EpisodicTrace):
            embedding = input_data.embedding
        else:
            embedding = np.array(input_data)
        
        if not self._initialized:
            return 0
        
        # Store in float16 by default
        embedding = embedding.astype(np.float16)
        # Convert to float32 only for kmeans operation
        embedding_f32 = embedding.astype(np.float32).reshape(1, -1)
        return self.kmeans.predict(embedding_f32)[0]
    
    def _update_cluster_stats(self, trace: EpisodicTrace, cluster_id: int):
        """Update statistics for a cluster."""
        self.trace_counts[cluster_id] += 1
        self.total_salience[cluster_id] += trace.salience
        
        # Create or update cluster summary
        if cluster_id not in self.summaries:
            self.summaries[cluster_id] = ClusterSummary(
                cluster_id=cluster_id,
                centroid=self.kmeans.cluster_centers_[cluster_id].astype(np.float16),
                size=1,
                avg_salience=trace.salience,
                creation_time=datetime.now(),
                last_updated=datetime.now(),
                keywords=self._extract_keywords(trace.text),
                summary_text=None
            )
        else:
            summary = self.summaries[cluster_id]
            summary.size = self.trace_counts[cluster_id]
            summary.avg_salience = (
                self.total_salience[cluster_id] / self.trace_counts[cluster_id]
            )
            summary.last_updated = datetime.now()
            summary.centroid = self.kmeans.cluster_centers_[cluster_id].astype(np.float16)
            
            # Update keywords (simple approach - could be more sophisticated)
            new_keywords = self._extract_keywords(trace.text)
            summary.keywords = list(set(summary.keywords + new_keywords))[:10]
    
    def _extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """Extract keywords from text (simple implementation)."""
        # This is a placeholder - in practice, you might want to use
        # more sophisticated keyword extraction (e.g., TF-IDF, RAKE, etc.)
        words = text.lower().split()
        # Remove common words and punctuation (very basic)
        words = [w for w in words if len(w) > 3 and w.isalnum()]
        return words[:max_keywords]
    
    def merge_small_clusters(self) -> List[Tuple[int, int]]:
        """Merge small clusters based on similarity."""
        merges = []
        small_clusters = [
            cid for cid, count in self.trace_counts.items()
            if count < self.config.min_cluster_size
        ]
        
        for small_cid in small_clusters:
            if small_cid not in self.summaries:
                continue
                
            # Find most similar larger cluster
            best_sim = -1
            best_target = -1
            small_centroid = self.summaries[small_cid].centroid
            
            for large_cid, summary in self.summaries.items():
                if large_cid == small_cid or large_cid in small_clusters:
                    continue
                    
                sim = cosine_similarity(small_centroid, summary.centroid)
                if sim > best_sim and sim > self.config.similarity_threshold:
                    best_sim = sim
                    best_target = large_cid
            
            if best_target != -1:
                merges.append((small_cid, best_target))
        
        return merges
    
    def get_similar_clusters(self, cluster_id: int, top_k: int = 3) -> List[Tuple[int, float]]:
        """Find most similar clusters to given cluster."""
        if cluster_id not in self.summaries:
            return []
        
        centroid = self.summaries[cluster_id].centroid
        similarities = []
        
        for cid, summary in self.summaries.items():
            if cid != cluster_id:
                sim = cosine_similarity(centroid, summary.centroid)
                similarities.append((cid, sim))
        
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k] 