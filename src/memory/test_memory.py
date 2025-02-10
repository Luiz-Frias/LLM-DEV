import numpy as np
from datetime import datetime, timedelta
import torch
import time

from .memory_components import EpisodicTrace, MonolithIntake, EpisodicMemory
from .hierarchical_memory import HierarchicalMemoryTree
from .reservoir import Reservoir

def create_dummy_trace(text: str, salience: float, embedding_dim: int = 768) -> EpisodicTrace:
    """Create a dummy trace for testing."""
    return EpisodicTrace(
        embedding=np.random.randn(embedding_dim).astype(np.float16),
        text=text,
        salience=salience
    )

def test_memory_system():
    print("1. Initializing memory components...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Initialize components
    memory_tree = HierarchicalMemoryTree(
        base_capacity=5,
        error_threshold=0.1,
        imbalance_threshold=2.0,
        device=device
    )
    
    episodic_memory = EpisodicMemory(
        input_dim=768,  # GPT-2 hidden size
        reservoir_dim=64,
        salience_threshold=0.6,
        spiking_threshold=0.7,
        device=device
    )
    
    monolith_intake = MonolithIntake(
        batch_size=2,
        device=device
    )
    
    print("\n2. Creating test traces...")
    test_texts = [
        # High salience group (technical)
        "Implementing neural network architecture with SSSM components",
        "Optimizing memory usage with hierarchical tree structures",
        "Integrating selective state space models for attention",
        
        # Medium salience group (documentation)
        "Writing documentation for memory components",
        "Updating implementation details for SSSM",
        "Adding code comments for memory optimization",
        
        # Low salience group (misc)
        "General discussion about AI",
        "Project planning notes",
        "Meeting summary"
    ]
    
    # Create embeddings (simulated)
    embeddings = [torch.randn(768) for _ in test_texts]
    
    print("\n3. Testing monolith intake...")
    for text, embedding in zip(test_texts, embeddings):
        monolith_intake.add_to_batch(text, embedding)
    
    traces = monolith_intake.process_batch()
    print(f"Processed traces: {len(traces)}")
    
    print("\n4. Testing episodic memory...")
    added_count = episodic_memory.add_traces(traces)
    print(f"Added traces: {added_count}/{len(traces)}")
    
    # Get memory statistics
    stats = episodic_memory.get_statistics()
    print("\nEpisodic Memory Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n5. Testing hierarchical tree...")
    # Add traces to tree
    for trace in traces:
        if isinstance(trace.embedding, np.ndarray):
            embedding = torch.from_numpy(trace.embedding)
        else:
            embedding = trace.embedding
        memory_tree.add_trace(embedding)
    
    # Get tree statistics
    tree_stats = memory_tree.get_statistics()
    print("\nHierarchical Tree Statistics:")
    for key, value in tree_stats.items():
        print(f"{key}: {value}")
    
    print("\n6. Testing memory pruning...")
    # Age some traces artificially
    for trace in traces[:2]:
        trace.timestamp = time.time() - 86400  # 24 hours ago
    
    # Prune old traces
    original_count = len(traces)
    episodic_memory.prune_old_traces(max_age_seconds=3600)  # 1 hour threshold
    final_count = len(episodic_memory.traces)
    print(f"Traces after pruning: {final_count}/{original_count}")
    
    print("\n7. Testing state serialization...")
    # Test state saving/loading
    tree_state = memory_tree.get_state()
    episodic_state = episodic_memory.get_state()
    monolith_state = monolith_intake.get_state()
    
    print("\nState sizes:")
    print(f"Tree state keys: {list(tree_state.keys())}")
    print(f"Episodic state keys: {list(episodic_state.keys())}")
    print(f"Monolith state keys: {list(monolith_state.keys())}")
    
    print("\nMemory system test completed successfully!")

if __name__ == "__main__":
    test_memory_system() 