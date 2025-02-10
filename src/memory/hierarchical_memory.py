import numpy as np
import torch
from typing import List, Dict, Optional
import time

def to_low_precision(vec, dtype=torch.float16):
    """Convert tensor to low precision."""
    if isinstance(vec, np.ndarray):
        vec = torch.from_numpy(vec)
    return vec.to(dtype)

def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two tensors with numerical stability."""
    # Convert to float32 for better numerical stability
    a_f = a.to(torch.float32)
    b_f = b.to(torch.float32)
    
    # Reshape if needed
    if a_f.dim() == 1:
        a_f = a_f.unsqueeze(0)
    if b_f.dim() == 1:
        b_f = b_f.unsqueeze(0)
    
    # Compute normalized dot product with eps for stability
    eps = 1e-8
    a_norm = torch.norm(a_f, dim=-1, keepdim=True).clamp_min(eps)
    b_norm = torch.norm(b_f, dim=-1, keepdim=True).clamp_min(eps)
    
    # Normalize vectors
    a_normalized = a_f / a_norm
    b_normalized = b_f / b_norm
    
    # Compute similarity
    similarity = torch.sum(a_normalized * b_normalized, dim=-1)
    
    return similarity.item()

def predictive_coding_update(new_embedding: torch.Tensor, predicted_embedding: torch.Tensor, error_threshold: float = 0.1) -> Optional[torch.Tensor]:
    """Apply predictive coding to determine if update is needed."""
    error = torch.norm(new_embedding - predicted_embedding)
    if error > error_threshold:
        return new_embedding - predicted_embedding
    return None

class HierarchicalMemoryNode:
    """A node in the hierarchical memory tree."""
    def __init__(self, embedding: torch.Tensor):
        self.centroid = to_low_precision(embedding)
        self.count = 1
        self.left = None
        self.right = None
        self.last_update = time.time()

    def update(self, new_embedding: torch.Tensor):
        """Update node centroid with new embedding."""
        self.centroid = to_low_precision(
            (self.centroid * self.count + new_embedding) / (self.count + 1)
        )
        self.count += 1
        self.last_update = time.time()

class HierarchicalMemoryTree:
    """
    Maintains episodic embeddings in a binary tree with adaptive splitting.
    Uses predictive coding to gate updates and merges overlapping leaves.
    """
    def __init__(
        self,
        base_capacity: int = 5,
        error_threshold: float = 0.1,
        imbalance_threshold: float = 2.0,
        device: torch.device = None
    ):
        self.root = None
        self.base_capacity = base_capacity
        self.error_threshold = error_threshold
        self.imbalance_threshold = imbalance_threshold
        self.device = device or torch.device("cpu")
        self.total_nodes = 0
        self.max_depth = 0

    def add_trace(self, embedding: torch.Tensor) -> bool:
        """Add a new trace to the tree, returns True if update was significant."""
        if self.root is None:
            self.root = HierarchicalMemoryNode(embedding)
            self.total_nodes = 1
            self.max_depth = 1
            return True
        return self._add_trace_recursive(self.root, embedding, depth=1)

    def _add_trace_recursive(self, node: HierarchicalMemoryNode, embedding: torch.Tensor, depth: int) -> bool:
        # Use predictive coding to determine if update is needed
        error_signal = predictive_coding_update(embedding, node.centroid, self.error_threshold)
        if error_signal is None:
            return False  # No significant update needed

        # Update node statistics
        node.update(embedding)
        self.max_depth = max(self.max_depth, depth)

        # Adaptive splitting based on capacity and imbalance
        if node.count >= self.base_capacity:
            if node.left is None and node.right is None:
                # Split leaf node
                perturb = torch.randn_like(node.centroid) * 0.01
                node.left = HierarchicalMemoryNode(node.centroid + perturb)
                node.right = HierarchicalMemoryNode(node.centroid - perturb)
                self.total_nodes += 2
            else:
                # Check for imbalance
                left_count = node.left.count if node.left else 0
                right_count = node.right.count if node.right else 0
                
                if left_count > 0 and right_count > 0:
                    ratio = max(left_count, right_count) / min(left_count, right_count)
                    if ratio >= self.imbalance_threshold:
                        # Route to less populated branch
                        target = node.left if left_count < right_count else node.right
                        return self._add_trace_recursive(target, embedding, depth + 1)

                # Route based on similarity
                sim_left = cosine_similarity(embedding, node.left.centroid)
                sim_right = cosine_similarity(embedding, node.right.centroid)
                if sim_left >= sim_right:
                    return self._add_trace_recursive(node.left, embedding, depth + 1)
                else:
                    return self._add_trace_recursive(node.right, embedding, depth + 1)

        return True

    def get_leaf_nodes(self) -> List[HierarchicalMemoryNode]:
        """Get all leaf nodes using an iterative approach."""
        leaves = []
        stack = [self.root] if self.root is not None else []
        while stack:
            node = stack.pop()
            if node.left is None and node.right is None:
                leaves.append(node)
            else:
                if node.left is not None:
                    stack.append(node.left)
                if node.right is not None:
                    stack.append(node.right)
        return leaves

    def merge_overlapping_leaves(self, similarity_threshold: float = 0.9) -> List[HierarchicalMemoryNode]:
        """Merge leaf nodes with similar centroids to reduce redundancy."""
        leaves = self.get_leaf_nodes()
        merged = []
        used = set()

        for i in range(len(leaves)):
            if i in used:
                continue
            node_i = leaves[i]
            merged_node = node_i

            for j in range(i + 1, len(leaves)):
                if j in used:
                    continue
                node_j = leaves[j]
                sim = cosine_similarity(merged_node.centroid, node_j.centroid)
                
                if sim >= similarity_threshold:
                    # Merge nodes by computing weighted centroid
                    total = merged_node.count + node_j.count
                    new_centroid = (
                        merged_node.centroid * merged_node.count + 
                        node_j.centroid * node_j.count
                    ) / total
                    merged_node.centroid = to_low_precision(new_centroid)
                    merged_node.count = total
                    merged_node.last_update = max(merged_node.last_update, node_j.last_update)
                    used.add(j)
                    self.total_nodes -= 1  # Account for merged node

            merged.append(merged_node)

        return merged

    def get_statistics(self) -> Dict:
        """Get tree statistics for monitoring."""
        return {
            "total_nodes": self.total_nodes,
            "max_depth": self.max_depth,
            "num_leaves": len(self.get_leaf_nodes())
        }

    def _node_to_dict(self, node: Optional[HierarchicalMemoryNode]) -> Optional[Dict]:
        """Convert a node to a serializable dictionary."""
        if node is None:
            return None
        return {
            'centroid': node.centroid.cpu().numpy().tolist(),
            'count': node.count,
            'last_update': node.last_update,
            'left': self._node_to_dict(node.left),
            'right': self._node_to_dict(node.right)
        }

    def _dict_to_node(self, data: Optional[Dict]) -> Optional[HierarchicalMemoryNode]:
        """Convert a dictionary back to a node."""
        if data is None:
            return None
        node = HierarchicalMemoryNode(
            torch.tensor(data['centroid'], device=self.device)
        )
        node.count = data['count']
        node.last_update = data['last_update']
        node.left = self._dict_to_node(data['left'])
        node.right = self._dict_to_node(data['right'])
        return node

    def get_state(self) -> Dict:
        """Get serializable state of the tree."""
        return {
            'base_capacity': self.base_capacity,
            'error_threshold': self.error_threshold,
            'imbalance_threshold': self.imbalance_threshold,
            'total_nodes': self.total_nodes,
            'max_depth': self.max_depth,
            'root': self._node_to_dict(self.root)
        }
    
    def set_state(self, state: Dict):
        """Set the state of the tree from a dictionary."""
        self.base_capacity = state['base_capacity']
        self.error_threshold = state['error_threshold']
        self.imbalance_threshold = state['imbalance_threshold']
        self.total_nodes = state['total_nodes']
        self.max_depth = state['max_depth']
        self.root = self._dict_to_node(state['root'])

    def rebalance_tree(self):
        """
        Rebalance the hierarchical memory tree to maintain efficient O(log n) operations.
        This version incorporates a lightweight predicted load metric per leaf to help guide sorting.
        The predicted load is computed as: load = node.count / (current_time - node.last_update + epsilon).
        Leaves are then sorted by a tuple (centroid[0], -load_metric) so that nodes with higher predicted load
        are grouped together, which may allow adaptive pruning to be more proactive.
        """
        # TODO: Dynamically adjust the heuristic weighting between the centroid value and the predicted load metric
        leaves = self.get_leaf_nodes()
        if not leaves:
            return
        
        import time
        current_time = time.time()
        epsilon = 1e-5
        
        # Compute load metrics for each leaf
        load_metrics = [leaf.count / (current_time - leaf.last_update + epsilon) for leaf in leaves]
        
        # Prepare sort keys: (centroid_first_value, -load_metric)
        sort_keys = [(leaf.centroid[0].item(), -load_metrics[i]) for i, leaf in enumerate(leaves)]
        
        # Sort leaves based on the tuple key; this is a heuristic combining spatial position and load
        sorted_leaves = [leaf for _, leaf in sorted(zip(sort_keys, leaves), key=lambda x: x[0])]

        # Recursive function to build a balanced binary tree from sorted leaves
        def build_balanced(leaves_list: List[HierarchicalMemoryNode]) -> HierarchicalMemoryNode:
            if len(leaves_list) == 1:
                return leaves_list[0]
            mid = len(leaves_list) // 2
            left_tree = build_balanced(leaves_list[:mid])
            right_tree = build_balanced(leaves_list[mid:])
            total_count = left_tree.count + right_tree.count
            new_centroid = (left_tree.centroid * left_tree.count + right_tree.centroid * right_tree.count) / total_count
            parent = HierarchicalMemoryNode(new_centroid)
            parent.count = total_count
            parent.left = left_tree
            parent.right = right_tree
            parent.last_update = max(left_tree.last_update, right_tree.last_update)
            return parent

        new_root = build_balanced(sorted_leaves)
        self.root = new_root
        self.total_nodes = len(leaves)

        # Compute the new maximum depth iteratively
        def compute_depth(node: Optional[HierarchicalMemoryNode]) -> int:
            if node is None:
                return 0
            return 1 + max(compute_depth(node.left), compute_depth(node.right))

        self.max_depth = compute_depth(self.root) 