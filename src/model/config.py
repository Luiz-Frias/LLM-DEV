from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """Configuration for GPT-2 model optimized for memory efficiency with SSSM."""
    
    # Model architecture
    model_name: str = "gpt2"  # Using smallest GPT-2 variant
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    
    # Memory optimization
    use_gradient_checkpointing: bool = True
    use_mixed_precision: bool = True  # Where supported by MPS
    
    # Training optimization
    batch_size: int = 2  # Hard cap at 2 for 8GB RAM
    max_length: int = 512
    gradient_accumulation_steps: int = 4
    
    # MPS (Metal Performance Shaders) settings
    use_mps: bool = True  # Use MPS backend when available
    fallback_to_cpu: bool = True  # Fallback to CPU if MPS is not available
    
    # SSSM Configuration
    use_ssmm: bool = True
    ssmm_state_size: int = 16
    spike_threshold: float = 0.1
    dt_min: float = 0.001
    dt_max: float = 0.1
    
    # Memory Components Configuration
    use_memory_components: bool = True
    tree_base_capacity: int = 5
    tree_error_threshold: float = 0.1
    tree_imbalance_threshold: float = 2.0
    episodic_reservoir_dim: int = 64
    episodic_salience_threshold: float = 0.6
    monolith_batch_size: int = 2
    
    def __post_init__(self):
        """Validate configuration for 8GB RAM setup."""
        if self.batch_size * self.max_length > 1024:  # Stricter limit with SSSM
            raise ValueError(
                f"batch_size * max_length = {self.batch_size * self.max_length} "
                f"exceeds memory constraints for 8GB RAM with SSSM"
            )
        
        if self.batch_size > 2:
            raise ValueError(
                f"batch_size = {self.batch_size} exceeds maximum of 2 "
                f"for 8GB RAM configuration"
            )
    
    def get_ssmm_config(self) -> dict:
        """Get SSSM configuration dictionary."""
        return {
            "state_size": self.ssmm_state_size,
            "spike_threshold": self.spike_threshold,
            "dt_min": self.dt_min,
            "dt_max": self.dt_max
        }
    
    def get_memory_config(self) -> dict:
        """Get memory components configuration dictionary."""
        return {
            "tree": {
                "base_capacity": self.tree_base_capacity,
                "error_threshold": self.tree_error_threshold,
                "imbalance_threshold": self.tree_imbalance_threshold
            },
            "episodic": {
                "reservoir_dim": self.episodic_reservoir_dim,
                "salience_threshold": self.episodic_salience_threshold
            },
            "monolith": {
                "batch_size": self.monolith_batch_size
            }
        }
