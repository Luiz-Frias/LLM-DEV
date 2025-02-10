from .attention import HybridAttention, GlobalMemoryAttention
from .optimization import (
    setup_memory_optimizations,
    setup_device,
    optimize_memory,
    setup_mixed_precision,
    get_memory_stats
)
from .config import ModelConfig

__all__ = [
    'HybridAttention',
    'GlobalMemoryAttention',
    'setup_memory_optimizations',
    'setup_device',
    'optimize_memory',
    'setup_mixed_precision',
    'get_memory_stats',
    'ModelConfig'
]

# Example usage:
"""
from src.model import ModelConfig, setup_memory_optimizations

# Create model configuration
config = ModelConfig(
    model_name="gpt2",
    batch_size=2,
    use_gradient_checkpointing=True,
    use_mixed_precision=True,
    use_mps=True
)

# Initialize model and optimize
model = AutoModelForCausalLM.from_pretrained("gpt2")
model, optimizer, memory_optimizer = setup_memory_optimizations(
    model=model,
    optimizer=optimizer,
    device=device,
    use_gradient_checkpointing=True
)

# Print model stats
print(f"Memory stats: {get_memory_stats()}")
"""
