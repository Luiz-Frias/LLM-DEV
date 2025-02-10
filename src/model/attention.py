import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
import math
import numpy as np

class SSMMLayer(nn.Module):
    """
    Selective State Space Model (SSSM) layer inspired by Mamba.
    Processes embeddings through a state space model with selective updates.
    """
    def __init__(
        self,
        hidden_size: int,
        state_size: int = 16,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        spike_threshold: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.spike_threshold = spike_threshold
        
        # State space parameters
        self.A = nn.Parameter(torch.randn(state_size, state_size) / math.sqrt(state_size))
        self.B = nn.Parameter(torch.randn(state_size, hidden_size) / math.sqrt(hidden_size))
        self.C = nn.Parameter(torch.randn(hidden_size, state_size) / math.sqrt(state_size))
        
        # Selective update gate
        self.update_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )
        
        # Initialize dt (time delta) parameters
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_scale = dt_scale
        if dt_init == "random":
            self.log_dt = nn.Parameter(torch.rand(hidden_size) * dt_scale)
        else:
            self.log_dt = nn.Parameter(torch.ones(hidden_size) * math.log(dt_init))
    
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with selective state updates.
        Args:
            x: Input tensor (batch_size, seq_length, hidden_size)
            state: Optional previous state
        Returns:
            output: Processed tensor
            new_state: Updated state
        """
        # TODO: Evaluate dynamic optimization of SSSM parameters, including selective update threshold and state space parameters, for improved efficiency.
        batch_size, seq_length, _ = x.size()
        
        # Initialize state if not provided
        if state is None:
            state = torch.zeros(
                batch_size, self.state_size,
                device=x.device, dtype=x.dtype
            )
        
        # Compute dt (time step size)
        dt = torch.exp(self.log_dt).clamp(self.dt_min, self.dt_max)
        
        # Initialize output tensor
        output = torch.zeros_like(x)
        new_state = state
        
        # Process sequence
        for t in range(seq_length):
            xt = x[:, t]
            dstate = torch.matmul(new_state, self.A.T) + torch.matmul(xt, self.B.T)
            update_gate = self.update_gate(xt)
            state_change = dstate * dt.unsqueeze(0)
            spike_mask = (torch.abs(state_change) > self.spike_threshold).float()
            state_change = state_change * spike_mask
            new_state = new_state + state_change * update_gate.unsqueeze(-1)
            output[:, t] = torch.matmul(new_state, self.C.T)
        
        return output, new_state

    def get_state(self) -> Dict:
        """Get serializable state of the SSMM layer."""
        return {
            'hidden_size': self.hidden_size,
            'state_size': self.state_size,
            'spike_threshold': self.spike_threshold,
            'dt_min': self.dt_min,
            'dt_max': self.dt_max,
            'dt_scale': self.dt_scale,
            'state_dict': {
                'A': self.A.detach().cpu().numpy(),
                'B': self.B.detach().cpu().numpy(),
                'C': self.C.detach().cpu().numpy(),
                'log_dt': self.log_dt.detach().cpu().numpy(),
                'update_gate': {
                    name: param.detach().cpu().numpy()
                    for name, param in self.update_gate.named_parameters()
                }
            }
        }
    
    def set_state(self, state: Dict):
        """Set the state of the SSMM layer from a dictionary."""
        self.hidden_size = state['hidden_size']
        self.state_size = state['state_size']
        self.spike_threshold = state['spike_threshold']
        self.dt_min = state['dt_min']
        self.dt_max = state['dt_max']
        self.dt_scale = state['dt_scale']
        
        # Load parameters
        state_dict = state['state_dict']
        self.A.data.copy_(torch.from_numpy(state_dict['A']))
        self.B.data.copy_(torch.from_numpy(state_dict['B']))
        self.C.data.copy_(torch.from_numpy(state_dict['C']))
        self.log_dt.data.copy_(torch.from_numpy(state_dict['log_dt']))
        
        # Load update gate parameters
        for name, param in self.update_gate.named_parameters():
            param.data.copy_(torch.from_numpy(state_dict['update_gate'][name]))

class HybridAttention(nn.Module):
    """
    Combines local sliding window attention with global memory-augmented attention
    and Mamba's SSSM for embedding processing.
    """
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        window_size: int = 128,
        num_global_tokens: int = 8,
        attention_dropout: float = 0.1,
        max_position_embeddings: int = 512,
        sparsity_threshold: float = 0.1,
        ssmm_state_size: int = 16,
        spike_threshold: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.window_size = window_size
        self.num_global_tokens = num_global_tokens
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sparsity_threshold = sparsity_threshold
        
        # Initialize sparse attention masks
        self.register_buffer(
            "sparse_mask",
            torch.zeros(1, num_attention_heads, 1, 1)
        )
        
        # Linear layers for local attention
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        # Linear layers for global tokens
        self.global_query = nn.Linear(hidden_size, self.all_head_size)
        self.global_key = nn.Linear(hidden_size, self.all_head_size)
        self.global_value = nn.Linear(hidden_size, self.all_head_size)
        
        # Output
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(attention_dropout)
        
        # Position embeddings for relative attention
        self.max_position_embeddings = max_position_embeddings
        self.position_embeddings = nn.Parameter(
            torch.randn(2 * max_position_embeddings - 1, self.attention_head_size)
        )
        
        # Add SSMM layer for embedding processing
        self.ssmm = SSMMLayer(
            hidden_size=hidden_size,
            state_size=ssmm_state_size,
            spike_threshold=spike_threshold
        )
        
        # State tracking for SSMM
        self.last_state = None
    
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Transpose and reshape tensor for attention computation."""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def get_local_attention_mask(
        self,
        batch_size: int,
        seq_length: int,
        device: torch.device
    ) -> torch.Tensor:
        """Create causal mask for local attention with sliding window."""
        # Create causal mask
        mask = torch.triu(
            torch.ones((seq_length, seq_length), device=device),
            diagonal=1
        ).bool()
        
        # Apply sliding window
        for i in range(seq_length):
            start = max(0, i - self.window_size)
            end = min(seq_length, i + self.window_size + 1)
            mask[i, start:end] = False
        
        # Expand for batch size and heads
        mask = mask.unsqueeze(0).unsqueeze(1)
        mask = mask.expand(batch_size, self.num_attention_heads, seq_length, seq_length)
        return mask
    
    def _validate_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Validate input tensor type and shape."""
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError(f"hidden_states must be torch.Tensor, got {type(hidden_states)}")
        if hidden_states.dim() != 3:
            raise ValueError(f"hidden_states must have 3 dimensions, got {hidden_states.dim()}")
        return hidden_states.to(self.position_embeddings.device)

    def _validate_global_memory(self, global_memory: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Validate global memory tensor if provided."""
        if global_memory is not None:
            if not isinstance(global_memory, torch.Tensor):
                raise TypeError(f"global_memory must be torch.Tensor, got {type(global_memory)}")
            if global_memory.dim() != 3:
                raise ValueError(f"global_memory must have 3 dimensions, got {global_memory.dim()}")
            return global_memory.to(self.position_embeddings.device)
        return None

    def _compute_sparse_attention_mask(
        self,
        attention_scores: torch.Tensor,
        sparsity_threshold: Optional[float] = None
    ) -> torch.Tensor:
        """Compute sparse attention mask based on score magnitudes."""
        threshold = sparsity_threshold or self.sparsity_threshold
        
        # Calculate attention magnitude
        attention_magnitudes = torch.abs(attention_scores)
        
        # Create mask for top-k values per query
        batch_size, num_heads, seq_len, _ = attention_scores.shape
        k = int((1 - threshold) * seq_len)  # Number of values to keep
        
        # Get top-k indices efficiently using torch.topk
        _, top_indices = torch.topk(attention_magnitudes, k, dim=-1)
        
        # Create sparse mask
        sparse_mask = torch.zeros_like(attention_scores, dtype=torch.bool)
        sparse_mask.scatter_(-1, top_indices, True)
        
        return sparse_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        global_memory: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Enhanced forward pass with SSMM processing."""
        hidden_states = self._validate_input(hidden_states)
        global_memory = self._validate_global_memory(global_memory)
        
        # Process through SSMM layer first
        ssmm_output, new_state = self.ssmm(hidden_states, self.last_state)
        self.last_state = new_state
        
        # Combine SSMM output with original hidden states using a residual connection
        hidden_states = hidden_states + ssmm_output
        
        batch_size, seq_length, _ = hidden_states.size()
        device = hidden_states.device
        
        # Process local attention with sparsity
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Get local attention mask
        local_attention_mask = self.get_local_attention_mask(batch_size, seq_length, device)
        if attention_mask is not None:
            local_attention_mask = local_attention_mask | attention_mask
        
        # Compute sparse attention mask
        sparse_mask = self._compute_sparse_attention_mask(attention_scores)
        
        # Apply both local and sparse masks
        combined_mask = local_attention_mask | ~sparse_mask
        attention_scores = attention_scores.masked_fill(combined_mask, float('-inf'))
        
        # Compute attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Use torch.sparse for the final matmul if sparsity is high
        if attention_probs.count_nonzero() / attention_probs.numel() < 0.3:
            # Convert to sparse tensor
            sparse_probs = attention_probs.to_sparse()
            local_context = torch.sparse.mm(
                sparse_probs.reshape(-1, sparse_probs.size(-1)),
                value_layer.reshape(-1, value_layer.size(-1))
            ).reshape(batch_size, self.num_attention_heads, seq_length, self.attention_head_size)
        else:
            local_context = torch.matmul(attention_probs, value_layer)
        
        # Process global memory if provided
        if global_memory is not None:
            global_query = self.transpose_for_scores(self.global_query(hidden_states))
            global_key = self.transpose_for_scores(self.global_key(global_memory))
            global_value = self.transpose_for_scores(self.global_value(global_memory))
            
            # Compute global attention scores with sparsity
            global_scores = torch.matmul(global_query, global_key.transpose(-1, -2))
            global_scores = global_scores / math.sqrt(self.attention_head_size)
            
            # Apply sparse attention to global memory
            global_sparse_mask = self._compute_sparse_attention_mask(
                global_scores,
                sparsity_threshold=self.sparsity_threshold * 0.5  # Less aggressive for global
            )
            global_scores = global_scores.masked_fill(~global_sparse_mask, float('-inf'))
            
            global_probs = F.softmax(global_scores, dim=-1)
            global_probs = self.dropout(global_probs)
            
            # Use sparse operations for global attention
            if global_probs.count_nonzero() / global_probs.numel() < 0.3:
                sparse_global_probs = global_probs.to_sparse()
                global_context = torch.sparse.mm(
                    sparse_global_probs.reshape(-1, sparse_global_probs.size(-1)),
                    global_value.reshape(-1, global_value.size(-1))
                ).reshape(batch_size, self.num_attention_heads, seq_length, self.attention_head_size)
            else:
                global_context = torch.matmul(global_probs, global_value)
            
            # Combine local and global context
            context_layer = local_context + global_context
        else:
            context_layer = local_context
        
        # Reshape output
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_shape)
        
        # Project to output size
        output = self.dense(context_layer)
        
        if output_attentions:
            return output, attention_probs
        return output, None

    def get_state(self) -> Dict:
        """Get serializable state including SSMM state and parameters."""
        state = {
            'hidden_size': self.hidden_size,
            'num_attention_heads': self.num_attention_heads,
            'window_size': self.window_size,
            'num_global_tokens': self.num_global_tokens,
            'max_position_embeddings': self.max_position_embeddings,
            'sparsity_threshold': self.sparsity_threshold,
            'position_embeddings': self.position_embeddings.detach().cpu().numpy(),
            'state_dict': {
                k: v.detach().cpu().numpy() 
                for k, v in self.state_dict().items()
                if k not in ['ssmm.A', 'ssmm.B', 'ssmm.C', 'ssmm.log_dt']  # Exclude SSMM params
            },
            'ssmm': self.ssmm.get_state(),  # Save SSMM state separately
            'last_state': self.last_state.detach().cpu().numpy() if self.last_state is not None else None
        }
        return state
    
    def set_state(self, state: Dict):
        """Set state including SSMM state and parameters."""
        self.hidden_size = state['hidden_size']
        self.num_attention_heads = state['num_attention_heads']
        self.window_size = state['window_size']
        self.num_global_tokens = state['num_global_tokens']
        self.max_position_embeddings = state['max_position_embeddings']
        self.sparsity_threshold = state['sparsity_threshold']
        self.position_embeddings = nn.Parameter(torch.from_numpy(state['position_embeddings']))
        
        # Load main state dict
        self.load_state_dict({
            k: torch.from_numpy(v) 
            for k, v in state['state_dict'].items()
        }, strict=False)  # Use strict=False since SSMM params are handled separately
        
        # Load SSMM state
        if 'ssmm' in state:
            self.ssmm.set_state(state['ssmm'])
        
        # Load last state if available
        if 'last_state' in state and state['last_state'] is not None:
            self.last_state = torch.from_numpy(state['last_state'])

class GlobalMemoryAttention(nn.Module):
    """
    Attention mechanism specifically for processing global memory tokens
    derived from the hierarchical memory tree and knowledge graph.
    """
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int = 8,
        attention_dropout: float = 0.1,
        max_global_tokens: int = 16
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.max_global_tokens = max_global_tokens
        
        # Projections for memory tokens
        self.memory_query = nn.Linear(hidden_size, hidden_size)
        self.memory_key = nn.Linear(hidden_size, hidden_size)
        self.memory_value = nn.Linear(hidden_size, hidden_size)
        
        # Output projection
        self.output = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(attention_dropout)
        
        # Layer norm for memory tokens
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def _validate_tensors(
        self,
        hidden_states: torch.Tensor,
        memory_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Validate input tensors."""
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError(f"hidden_states must be torch.Tensor, got {type(hidden_states)}")
        if not isinstance(memory_tokens, torch.Tensor):
            raise TypeError(f"memory_tokens must be torch.Tensor, got {type(memory_tokens)}")
        
        if hidden_states.dim() != 3:
            raise ValueError(f"hidden_states must have 3 dimensions, got {hidden_states.dim()}")
        if memory_tokens.dim() != 3:
            raise ValueError(f"memory_tokens must have 3 dimensions, got {memory_tokens.dim()}")
        
        device = self.memory_query.weight.device
        return hidden_states.to(device), memory_tokens.to(device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        hidden_states, memory_tokens = self._validate_tensors(hidden_states, memory_tokens)
        """
        Process input hidden states with global memory tokens.
        
        Args:
            hidden_states: (batch_size, seq_length, hidden_size)
            memory_tokens: (batch_size, num_memory_tokens, hidden_size)
            attention_mask: Optional mask for memory tokens
        """
        batch_size, seq_length, _ = hidden_states.size()
        
        # Project memory tokens
        memory_tokens = self.layer_norm(memory_tokens)
        memory_q = self.memory_query(hidden_states)
        memory_k = self.memory_key(memory_tokens)
        memory_v = self.memory_value(memory_tokens)
        
        # Reshape for attention
        memory_q = memory_q.view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size)
        memory_k = memory_k.view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
        memory_v = memory_v.view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
        
        # Transpose for attention computation
        memory_q = memory_q.transpose(1, 2)
        memory_k = memory_k.transpose(1, 2)
        memory_v = memory_v.transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(memory_q, memory_k.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask, float('-inf'))
        
        # Apply softmax and dropout
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Compute context layer
        context = torch.matmul(attention_probs, memory_v)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_length, self.hidden_size)
        
        # Project to output size
        output = self.output(context)
        return output 

    def get_state(self) -> Dict:
        """Get serializable state of the memory attention module."""
        return {
            'hidden_size': self.hidden_size,
            'num_attention_heads': self.num_attention_heads,
            'max_global_tokens': self.max_global_tokens,
            'state_dict': {
                k: v.detach().cpu().numpy()
                for k, v in self.state_dict().items()
            }
        }
    
    def set_state(self, state: Dict):
        """Set the state of the memory attention module from a dictionary."""
        self.hidden_size = state['hidden_size']
        self.num_attention_heads = state['num_attention_heads']
        self.max_global_tokens = state['max_global_tokens']
        self.load_state_dict({
            k: torch.from_numpy(v)
            for k, v in state['state_dict'].items()
        }) 