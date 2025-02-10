import numpy as np
import time
from typing import List, Optional, Dict, Union
from sklearn.cluster import MiniBatchKMeans
import torch
from dataclasses import dataclass
from .clustering import DynamicClusterer  # Import from clustering module
from .hierarchical_memory import HierarchicalMemoryTree
from .reservoir import Reservoir

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
    """A single memory trace containing embedded information and metadata."""
    embedding: np.ndarray
    text: str
    salience: float
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if isinstance(self.embedding, torch.Tensor):
            self.embedding = self.embedding.detach().cpu().numpy()
        if not isinstance(self.embedding, np.ndarray):
            raise TypeError(f"Embedding must be a numpy array, got {type(self.embedding)}")
        self.embedding = to_low_precision(self.embedding)

    def should_spike(self, threshold: float = 0.7) -> bool:
        return self.salience >= threshold

class EphemeralMemory:
    """Short-term memory buffer for recent inputs."""
    
    def __init__(self):
        self.current_input = None
    
    def load_input(self, text: str, embedding: torch.Tensor):
        """Load new input into ephemeral memory."""
        self.current_input = {
            'text': text,
            'embedding': embedding.detach() if isinstance(embedding, torch.Tensor) else embedding
        }

    def get_recent_inputs(self, k: int = 5) -> List[dict]:
        """Get k most recent inputs from buffer."""
        return [self.current_input]

    def clear(self):
        """Clear current input and buffer."""
        self.current_input = None

    def get_state(self) -> Dict:
        """Get serializable state of the ephemeral memory."""
        return {
            'current_input': self.current_input
        }
    
    def set_state(self, state: Dict):
        """Set the state of the ephemeral memory from a dictionary."""
        self.current_input = state['current_input']

class EpisodicMemory:
    """Long-term episodic memory storage with salience-based filtering."""
    def __init__(
        self,
        input_dim: int = 768,  # Base model hidden size
        reservoir_dim: int = 64,
        salience_threshold: float = 0.6,
        spiking_threshold: float = 0.7,
        tree_capacity: int = 5,
        error_threshold: float = 0.1,
        imbalance_threshold: float = 2.0,
        device: Optional[torch.device] = None
    ):
        self.device = device or torch.device("cpu")
        self.salience_threshold = salience_threshold
        self.spiking_threshold = spiking_threshold
        
        # Initialize hierarchical memory tree
        self.tree = HierarchicalMemoryTree(
            base_capacity=tree_capacity,
            error_threshold=error_threshold,
            imbalance_threshold=imbalance_threshold,
            device=self.device
        )
        
        # Initialize reservoir for stable projections
        self.reservoir = Reservoir(
            input_dim=input_dim,
            reservoir_dim=reservoir_dim,
            device=self.device
        )
        
        # Keep track of traces and their projections
        self.traces: List[EpisodicTrace] = []
        self.projected_traces: Dict[int, torch.Tensor] = {}
    
    def add_trace(self, trace: EpisodicTrace) -> bool:
        """Add a trace to memory if it meets salience criteria."""
        if trace.salience >= self.salience_threshold and trace.should_spike(self.spiking_threshold):
            # Project embedding through reservoir
            # Note: Reservoir now handles numpy arrays internally
            with torch.no_grad():
                projected = self.reservoir(trace.embedding)
                projected = projected.to(torch.float16)  # Convert to float16 for memory efficiency
            
            # Add to hierarchical tree and check if update was significant
            was_significant = self.tree.add_trace(projected)
            
            if was_significant:
                trace_id = len(self.traces)
                self.traces.append(trace)
                self.projected_traces[trace_id] = projected
                return True
        return False
    
    def add_traces(self, traces: List[EpisodicTrace]) -> int:
        """Add multiple traces to memory."""
        added_count = 0
        for trace in traces:
            if self.add_trace(trace):
                added_count += 1
        return added_count
    
    def get_merged_traces(self, similarity_threshold: float = 0.9) -> List[torch.Tensor]:
        """Get merged leaf node embeddings from the tree."""
        merged_leaves = self.tree.merge_overlapping_leaves(similarity_threshold)
        return [node.centroid for node in merged_leaves]
    
    def get_statistics(self) -> Dict:
        """Get memory statistics."""
        tree_stats = self.tree.get_statistics()
        return {
            "num_traces": len(self.traces),
            "num_projected": len(self.projected_traces),
            **tree_stats
        }
    
    def prune_old_traces(self, max_age_seconds: float = 86400, min_salience: float = 0.6):
        """Remove old traces unless they have high salience."""
        now = time.time()
        keep_indices = []
        
        for i, trace in enumerate(self.traces):
            if (now - trace.timestamp) < max_age_seconds or trace.salience > min_salience:
                keep_indices.append(i)
        
        # Update traces and projections
        self.traces = [self.traces[i] for i in keep_indices]
        self.projected_traces = {
            new_idx: self.projected_traces[old_idx]
            for new_idx, old_idx in enumerate(keep_indices)
        }
        
        # Rebuild tree with remaining traces
        self.tree = HierarchicalMemoryTree(
            base_capacity=self.tree.base_capacity,
            error_threshold=self.tree.error_threshold,
            imbalance_threshold=self.tree.imbalance_threshold,
            device=self.device
        )
        
        for projected in self.projected_traces.values():
            self.tree.add_trace(projected)

    def get_state(self) -> Dict:
        """Get serializable state of the episodic memory."""
        return {
            'traces': [
                {
                    'embedding': trace.embedding.tolist(),  # NumPy array to list
                    'text': trace.text,
                    'salience': trace.salience,
                    'timestamp': trace.timestamp
                }
                for trace in self.traces
            ],
            'salience_threshold': self.salience_threshold,
            'spiking_threshold': self.spiking_threshold,
            'tree_capacity': self.tree.base_capacity,
            'error_threshold': self.tree.error_threshold,
            'imbalance_threshold': self.tree.imbalance_threshold,
            'projected_traces': {
                idx: projected.tolist()
                for idx, projected in self.projected_traces.items()
            }
        }
    
    def set_state(self, state: Dict):
        """Set the state of the episodic memory from a dictionary."""
        self.salience_threshold = state['salience_threshold']
        self.spiking_threshold = state['spiking_threshold']
        self.tree.base_capacity = state['tree_capacity']
        self.tree.error_threshold = state['error_threshold']
        self.tree.imbalance_threshold = state['imbalance_threshold']
        self.traces = [
            EpisodicTrace(
                embedding=np.array(trace['embedding'], dtype=np.float16),  # List to NumPy array
                text=trace['text'],
                salience=trace['salience'],
                timestamp=trace['timestamp']
            )
            for trace in state['traces']
        ]
        self.projected_traces = {
            idx: torch.tensor(projected)
            for idx, projected in state['projected_traces'].items()
        }

    def get_embeddings_tensor(self) -> torch.Tensor:
        """Get all trace embeddings as a single tensor."""
        if not self.traces:
            return torch.empty(0, device=self.device)
        
        # Stack embeddings and ensure they're float16 tensors
        embeddings = []
        for trace in self.traces:
            if isinstance(trace.embedding, np.ndarray):
                emb = torch.from_numpy(trace.embedding)
            else:
                emb = trace.embedding
            embeddings.append(emb.to(torch.float16))
        
        # Stack and move to device
        stacked = torch.stack(embeddings).to(self.device)
        return stacked

class MonolithIntake:
    """Batch processor for inputs inspired by TikTok's monolith algo."""
    def __init__(self, 
                 batch_size: int = 32,
                 device: torch.device = None):
        self.batch_size = batch_size
        self.device = device or torch.device('cpu')
        self.current_batch = []

    def _validate_embedding(self, embedding: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Validate and convert embedding to correct type and device."""
        if isinstance(embedding, np.ndarray):
            embedding = torch.from_numpy(embedding)
        elif not isinstance(embedding, torch.Tensor):
            raise TypeError(f"Embedding must be torch.Tensor or np.ndarray, got {type(embedding)}")
        
        # Ensure float16 precision
        embedding = embedding.to(dtype=torch.float16, device=self.device)
        return embedding

    def add_to_batch(self, text: str, embedding: Union[torch.Tensor, np.ndarray]):
        """Add item to current batch with type validation."""
        embedding = self._validate_embedding(embedding)
        self.current_batch.append({
            'text': text,
            'embedding': embedding
        })

    def process_batch(self) -> List[EpisodicTrace]:
        """Process current batch and convert to EpisodicTraces."""
        if not self.current_batch:
            return []
        
        traces = []
        
        # Calculate average embedding for reference
        embeddings = torch.stack([item['embedding'] for item in self.current_batch])
        avg_embedding = embeddings.mean(dim=0)
        
        for item in self.current_batch:
            embedding = item['embedding']
            # Calculate salience based on distance from mean and text properties
            if isinstance(embedding, torch.Tensor):
                embedding_np = embedding.detach().cpu().numpy()
            else:
                embedding_np = embedding
            
            # Calculate salience based on:
            # 1. Distance from mean (normalized)
            # 2. Text length (normalized)
            # 3. Text complexity (basic measure)
            
            # Distance from mean (cosine similarity)
            distance = 1 - torch.nn.functional.cosine_similarity(
                embedding.unsqueeze(0),
                avg_embedding.unsqueeze(0)
            ).item()
            
            # Text length normalization (sigmoid-like scaling)
            text_len = len(item['text'].split())
            length_factor = min(1.0, text_len / 100)  # Cap at 100 words
            
            # Combine factors for final salience
            salience = (0.7 * distance + 0.3 * length_factor)
            salience = max(0.1, min(0.9, salience))  # Bound between 0.1 and 0.9
            
            traces.append(EpisodicTrace(
                text=item['text'],
                embedding=embedding_np,
                salience=salience,
                timestamp=time.time()
            ))
        
        self.current_batch = []  # Clear batch after processing
        return traces

    def compute_batch_statistics(self, traces: List[EpisodicTrace]) -> dict:
        """Compute statistics over a batch of traces."""
        if not traces:
            return {}
            
        # Ensure all embeddings are numpy arrays
        embeddings = np.stack([
            t.embedding if isinstance(t.embedding, np.ndarray)
            else np.array(t.embedding, dtype=np.float16)
            for t in traces
        ])
        saliences = np.array([t.salience for t in traces])
        
        return {
            'mean_salience': float(saliences.mean()),
            'std_salience': float(saliences.std()),
            'mean_embedding_norm': float(np.linalg.norm(embeddings, axis=1).mean()),
            'num_traces': len(traces)
        }

    def get_state(self) -> Dict:
        """Get serializable state of the monolith intake."""
        return {
            'batch_size': self.batch_size,
            'current_batch': [
                {
                    'text': item['text'],
                    'embedding': item['embedding'].detach().cpu().numpy() 
                    if isinstance(item['embedding'], torch.Tensor)
                    else item['embedding']
                }
                for item in self.current_batch
            ]
        }
    
    def set_state(self, state: Dict):
        """Set the state of the monolith intake from a dictionary."""
        self.batch_size = state['batch_size']
        self.current_batch = [
            {
                'text': item['text'],
                'embedding': torch.from_numpy(item['embedding'])
                if isinstance(item['embedding'], np.ndarray)
                else item['embedding']
            }
            for item in state['current_batch']
        ] 