import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque
from typing import Optional, Tuple, List, Dict
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, hidden_size: int, action_size: int, dropout: float = 0.1, sssm: Optional[nn.Module] = None):
        super(PolicyNetwork, self).__init__()
        self.sssm = sssm
        self.network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, action_size)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the policy network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, hidden_size)
            
        Returns:
            torch.Tensor: Action logits of shape (batch_size, action_size)
        """
        # If an SSSM module is provided, apply it with residual connection
        if self.sssm is not None:
            original_x = x
            x_seq = x.unsqueeze(1)  # shape: (batch, 1, hidden_size)
            x_transformed, _ = self.sssm(x_seq)
            x = x_transformed.squeeze(1) + original_x
        return self.network(x)

class ValueNetwork(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1, sssm: Optional[nn.Module] = None):
        super(ValueNetwork, self).__init__()
        self.sssm = sssm
        self.network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the value network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, hidden_size)
            
        Returns:
            torch.Tensor: Value predictions of shape (batch_size, 1)
        """
        if self.sssm is not None:
            original_x = x
            x_seq = x.unsqueeze(1)  # shape: (batch, 1, hidden_size)
            x_transformed, _ = self.sssm(x_seq)
            x = x_transformed.squeeze(1) + original_x
        return self.network(x)

class ReplayBuffer:
    """Memory buffer for experience replay."""
    
    def __init__(self, max_size: int = 10000, device: torch.device = None):
        self.max_size = max_size
        self.device = device or torch.device('cpu')
        self.buffer = []
    
    def get_state(self) -> Dict:
        """Get serializable state of the replay buffer."""
        return {
            'max_size': self.max_size,
            'buffer': [
                (
                    state.cpu().numpy().tolist(),
                    action.cpu().numpy().tolist(),
                    reward.cpu().numpy().tolist(),
                    next_state.cpu().numpy().tolist(),
                    done.cpu().numpy().tolist(),
                    state_embed.cpu().numpy().tolist() if state_embed is not None else None,
                    next_state_embed.cpu().numpy().tolist() if next_state_embed is not None else None
                )
                for state, action, reward, next_state, done, state_embed, next_state_embed in self.buffer
            ]
        }

    def set_state(self, state: Dict):
        """Set the state of the replay buffer from a dictionary."""
        self.buffer.clear()
        for transition in state['buffer']:
            state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor, state_embed_tensor, next_state_embed_tensor = transition
            self.buffer.append((
                torch.tensor(state_tensor, device=self.device),
                torch.tensor(action_tensor, device=self.device),
                torch.tensor(reward_tensor, device=self.device),
                torch.tensor(next_state_tensor, device=self.device),
                torch.tensor(done_tensor, device=self.device),
                torch.tensor(state_embed_tensor, device=self.device) if state_embed_tensor is not None else None,
                torch.tensor(next_state_embed_tensor, device=self.device) if next_state_embed_tensor is not None else None
            ))

def calculate_reward(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    pred_embed: Optional[torch.Tensor] = None,
    target_embed: Optional[torch.Tensor] = None,
    alpha: float = 0.5
) -> torch.Tensor:
    """
    Calculate the reward using a DeepSeek R1-style approach.
    
    The reward is computed as:
        reward = cosine_similarity(pred_embed, target_embed) - α * MSE(predictions, targets)
    If embeddings are not provided:
        reward = 1.0 - α * MSE(predictions, targets)
    
    Args:
        predictions (torch.Tensor): The predicted outputs
        targets (torch.Tensor): The target outputs
        pred_embed (torch.Tensor, optional): Embeddings for predictions
        target_embed (torch.Tensor, optional): Embeddings for targets
        alpha (float): Hyperparameter to adjust penalty strength (default: 0.5)
    
    Returns:
        torch.Tensor: Reward tensor per example
    """
    # Compute MSE loss (per example)
    mse_loss = F.mse_loss(predictions, targets, reduction="none").mean(dim=-1)
    
    if pred_embed is not None and target_embed is not None:
        # Normalize embeddings
        pred_embed = F.normalize(pred_embed, p=2, dim=-1)
        target_embed = F.normalize(target_embed, p=2, dim=-1)
        
        # Compute cosine similarity per example
        cos_sim = F.cosine_similarity(pred_embed, target_embed, dim=-1)
        
        # Combine similarity with MSE penalty
        reward = cos_sim - alpha * mse_loss
    else:
        # Without embeddings, use simple reward with MSE penalty
        reward = 1.0 - alpha * mse_loss
    
    return reward

class PPOMemory:
    """
    Memory buffer for PPO training, storing trajectories.
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.embeddings = []
    
    def store_memory(self, state, action, probs, vals, reward, done, embedding=None):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)
        self.embeddings.append(embedding)
    
    def clear_memory(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.embeddings = []
    
    def generate_batches(self, batch_size: int) -> List[np.ndarray]:
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+batch_size] for i in batch_start]
        return batches

    def get_state(self) -> Dict:
        """Get serializable state of the PPO memory."""
        return {
            'states': [state.cpu().numpy().tolist() for state in self.states],
            'actions': [action.cpu().numpy().tolist() for action in self.actions],
            'probs': [prob.cpu().numpy().tolist() for prob in self.probs],
            'vals': [val.cpu().numpy().tolist() for val in self.vals],
            'rewards': self.rewards,
            'dones': self.dones,
            'embeddings': [emb.cpu().numpy().tolist() if emb is not None else None for emb in self.embeddings]
        }
    
    def set_state(self, state: Dict):
        """Set the state of the PPO memory from a dictionary."""
        self.states = [torch.tensor(s) for s in state['states']]
        self.actions = [torch.tensor(a) for a in state['actions']]
        self.probs = [torch.tensor(p) for p in state['probs']]
        self.vals = [torch.tensor(v) for v in state['vals']]
        self.rewards = state['rewards']
        self.dones = state['dones']
        self.embeddings = [torch.tensor(e) if e is not None else None for e in state['embeddings']]

def compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    gamma: float = 0.99,
    gae_lambda: float = 0.95
) -> Tuple[List[float], List[float]]:
    """
    Compute Generalized Advantage Estimation (GAE).
    
    Args:
        rewards: List of rewards
        values: List of value estimates
        dones: List of done flags
        gamma: Discount factor
        gae_lambda: GAE smoothing parameter
    
    Returns:
        Tuple of advantages and returns
    """
    advantages = []
    returns = []
    gae = 0
    
    for t in reversed(range(len(rewards)-1)):
        delta = rewards[t] + gamma * values[t+1] * (1-dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1-dones[t]) * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + values[t])
    
    return advantages, returns 