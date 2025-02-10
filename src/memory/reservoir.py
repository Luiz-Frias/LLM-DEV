import torch
import torch.nn as nn
from typing import Optional, Dict, Union
import numpy as np

class Reservoir(nn.Module):
    """
    Projects input embeddings into a fixed, nonlinear state space.
    Uses a fixed projection size to ensure stable representations.
    """
    def __init__(
        self,
        input_dim: int,
        reservoir_dim: int = 64,
        activation: str = "tanh",
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.reservoir_dim = reservoir_dim
        self.device = device or torch.device("cpu")
        
        # Initialize fixed projection matrix
        self.projection = nn.Linear(input_dim, reservoir_dim, bias=False)
        
        # Initialize with orthogonal weights for better stability
        nn.init.orthogonal_(self.projection.weight)
        
        # Freeze the weights to maintain fixed projection
        for param in self.parameters():
            param.requires_grad = False
            
        # Set activation function
        self.activation = getattr(torch, activation)
        
        self.to(self.device)
    
    def forward(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Project input to reservoir space."""
        # Convert numpy array to tensor if needed
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        
        # Ensure input is float32 for stable projection
        x = x.to(torch.float32)
        
        # Ensure input has right shape
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        # Move to correct device
        x = x.to(self.device)
        
        # Project and apply nonlinearity
        projected = self.projection(x)
        return self.activation(projected)
    
    def get_state_dim(self) -> int:
        """Return dimension of reservoir state space."""
        return self.reservoir_dim

    def get_state(self) -> Dict:
        """Get serializable state of the reservoir."""
        return {
            'input_dim': self.input_dim,
            'reservoir_dim': self.reservoir_dim,
            'projection_weights': self.projection.weight.cpu().numpy().tolist(),
            'activation': self.activation.__name__
        }
    
    def set_state(self, state: Dict):
        """Set the state of the reservoir from a dictionary."""
        self.input_dim = state['input_dim']
        self.reservoir_dim = state['reservoir_dim']
        weights = torch.tensor(state['projection_weights'])
        self.projection.weight.data.copy_(weights)
        self.activation = getattr(torch, state['activation']) 