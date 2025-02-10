# Model Card: Memory-Efficient GPT-2 with SSSM

## Model Overview
- **Base Model**: GPT-2 (small variant)
- **Parameters**: 124M
- **Target Hardware**: Apple Silicon M2 with 8GB RAM
- **Primary Focus**: Memory-efficient training and inference with advanced memory components
- **License**: MIT

## Architecture Details

### Base Architecture
- GPT-2 small configuration
- 12 attention heads
- 768 hidden dimension
- 50,257 vocabulary size
- Mixed precision training (float16)

### Memory Optimizations
1. **Selective State Space Model (SSSM)**
   - Continuous-time state space modeling
   - Selective update mechanism
   - Spike-based thresholding
   - Efficient state tracking

2. **Hierarchical Memory Tree**
   - Adaptive node splitting
   - O(log n) operations
   - Predictive load balancing
   - Automatic rebalancing

3. **Monolith Intake System**
   - Efficient batch processing
   - Salience-based filtering
   - Type-safe operations
   - Memory-aware batching

4. **Memory Management**
   - Float16 precision tensors
   - Device-aware placement
   - Automatic garbage collection
   - Memory-mapped caching

## Hardware Requirements

### Minimum Specifications
- Apple Silicon M2
- 8GB Unified Memory
- macOS Sonoma or newer
- 512GB SSD (recommended)

### Memory Usage
- Model Parameters: ~500MB
- Training Peak: ~7GB RAM
- Inference Peak: 2-3GB RAM
- Disk Cache: ~1GB

## Performance Characteristics

### Training
- **Batch Size**: 2 (hard cap)
- **Gradient Accumulation**: 4 steps
- **Learning Rate**: 2e-5 to 5e-5
- **Sequence Length**: 512 tokens
- **Training Time**: ~24 hours on M2

### Inference
- **Latency**: 50-100ms per generation
- **Memory Usage**: 2-3GB RAM
- **Throughput**: 10-20 requests/second
- **Context Window**: 512 tokens

## Memory Management Features

### 1. SSSM Integration
- State space parameter optimization
- Selective update gating
- Spike-based memory updates
- Continuous-time dynamics

### 2. Hierarchical Memory
- Adaptive tree structure
- Efficient node operations
- Predictive load metrics
- Automatic pruning

### 3. Caching Strategy
- Memory-mapped datasets
- Efficient streaming
- Salience-based retention
- Dynamic cache sizing

## Known Limitations

1. **Hardware Constraints**
   - 8GB RAM ceiling
   - Limited by unified memory
   - Batch size restrictions
   - MPS backend limitations

2. **Training Constraints**
   - Maximum batch size of 2
   - Fixed sequence length
   - Limited model size options
   - Memory-speed tradeoffs

3. **Inference Limitations**
   - Context window restrictions
   - Memory-bound generation
   - Limited parallel processing
   - Device-specific constraints

## Future Development

### Short-term Improvements
1. **Memory Efficiency**
   - Enhanced SSSM parameters
   - Better tree rebalancing
   - Improved cache utilization
   - Reduced fragmentation

2. **Performance**
   - Optimized state tracking
   - Faster token generation
   - Better MPS utilization
   - Reduced memory overhead

3. **Training Stability**
   - Improved memory monitoring
   - Better error handling
   - Enhanced numerical stability
   - Robust checkpointing

### Long-term Goals
1. **Model Scaling**
   - Support for larger variants
   - Enhanced memory efficiency
   - Better hardware utilization
   - Advanced optimizations

2. **Hardware Support**
   - Improved MPS acceleration
   - Better memory management
   - Enhanced compatibility
   - Optimized performance

## Usage Guidelines

### Training Configuration
```python
config = ModelConfig(
    model_name="gpt2",
    batch_size=2,
    gradient_accumulation_steps=4,
    use_gradient_checkpointing=True,
    use_mixed_precision=True,
    use_mps=True,
    ssmm_state_size=16,
    spike_threshold=0.1
)
```

### Memory Monitoring
```python
from src.model.optimization import get_memory_stats

# Monitor memory usage
stats = get_memory_stats()
print(f"Memory usage: {stats}")
```

## Training Pipeline

### Dataset Configuration
- **Primary Dataset**: C4 (Colossal Clean Crawled Corpus)
- **Language**: English
- **Processing**: Cleaned and filtered
- **Streaming**: Memory-efficient loading
- **Cache Strategy**: Memory-mapped files

### Training Process
1. **Data Loading**
   - Streaming implementation
   - Dynamic batch sampling
   - Memory-mapped caching
   - Efficient collation

2. **Optimization Strategy**
   - AdamW optimizer
   - Linear learning rate schedule
   - Mixed precision (float16)
   - Gradient accumulation (4 steps)

3. **Memory Management**
   - SSSM state tracking
   - Hierarchical tree updates
   - Real-time monitoring
   - Automatic optimization

4. **Training Loop**
   - Early stopping
   - Gradient clipping
   - Checkpoint management
   - Memory-efficient validation

### Training Monitoring
```python
# Training configuration example
train_config = {
    "dataset": "c4",
    "dataset_subset": "en",
    "max_length": 512,
    "batch_size": 2,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-5,
    "warmup_steps": 1000,
    "max_steps": 100000,
    "save_steps": 10000,
    "eval_steps": 5000,
    "ssmm_config": {
        "state_size": 16,
        "spike_threshold": 0.1
    }
}
```

## Citation

```bibtex
@software{memory_efficient_gpt2_sssm,
  title={Memory-Efficient GPT-2 Implementation with SSSM},
  author={Luiz Frias},
  year={2024},
  url={https://github.com/Luiz-Frias/LLM-Dev}
}
```

## Additional Information
- **Documentation**: See repository README
- **Examples**: Available in `examples/` directory
- **Support**: GitHub issues and discussions
- **Updates**: Regular maintenance and optimization updates 