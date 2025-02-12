# Promethean Context Engine

A memory-efficient language model that brings advanced AI capabilities to resource-constrained environments. Like its namesake Prometheus who brought fire to humanity, this engine brings large language model capabilities to devices with limited memory through innovative memory management and contextual understanding.

## Key Features

### 1. Promethean Memory Architecture
- **Selective State Space Model (SSSM)**: Foresight-driven state space modeling with selective updates
- **Hierarchical Memory Tree**: Efficient O(log n) operations with adaptive splitting
- **Monolith Intake System**: Memory-aware batch processing with predictive filtering
- **Memory-Mapped Caching**: Efficient disk-based data streaming

### 2. Resource Optimization
- Mixed precision training (float16)
- Device-aware tensor placement
- Dynamic batch sizing (max 2)
- Gradient accumulation (4 steps)

### 3. Adaptive Learning
- Phase 1: C4 dataset for foundational understanding
- Phase 2: Chain-of-Thought fine-tuning with GSM8K and StrategyQA
- Synthetic data integration with adaptive ratios

## Architecture Overview

### Memory Components

#### 1. Selective State Space Model (SSSM)
- Predictive state space optimization
- Selective update gating
- Spike-based memory updates
- Continuous-time dynamics

#### 2. Hierarchical Memory Tree
- Adaptive node splitting
- Predictive load balancing
- Automatic rebalancing
- Efficient retrieval operations

#### 3. Monolith Intake
- Batch processing with type safety
- Salience-based filtering
- Memory-aware operations
- Efficient state tracking

### Performance Metrics
- Training Memory: 6-7GB peak
- Inference Memory: 2-3GB peak
- Throughput: 10-20 requests/s
- Latency: 50-100ms per generation

## Getting Started

### Prerequisites
```bash
# Hardware Requirements
- Apple Silicon M2 or newer
- 8GB RAM minimum
- macOS Sonoma or newer
- 512GB SSD recommended

# Software Requirements
- Python 3.9+
- PyTorch 2.0+
- transformers 4.30+
```

### Installation
```bash
# Clone the repository
git clone https://github.com/Luiz-Frias/LLM-Dev.git
cd LLM-Dev

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

#### Training
```python
from src.training.train_rl import train
from src.model.config import ModelConfig

# Configure training
config = ModelConfig(
    model_name="gpt2",
    batch_size=2,
    gradient_accumulation_steps=4,
    use_mixed_precision=True,
    use_mps=True,
    ssmm_state_size=16,
    spike_threshold=0.1
)

# Start training
train(config)
```

#### Inference
```python
from src.serve.api_server import app
import uvicorn

# Run API server
uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Memory Management

### 1. SSSM Configuration
```python
ssmm_config = {
    "state_size": 16,
    "spike_threshold": 0.1,
    "dt_min": 0.001,
    "dt_max": 0.1
}
```

### 2. Memory Tree Settings
```python
tree_config = {
    "base_capacity": 5,
    "error_threshold": 0.1,
    "imbalance_threshold": 2.0
}
```

### 3. Monolith Intake
```python
intake_config = {
    "batch_size": 2,
    "salience_threshold": 0.6
}
```

## Training Pipeline

### 1. Data Loading
- Streaming dataset implementation
- Memory-mapped caching
- Dynamic batch sampling
- Efficient collation

### 2. Training Loop
- SSSM state tracking
- Tree rebalancing
- Memory monitoring
- Checkpoint management

### 3. Memory Optimization
- Automatic garbage collection
- Device-aware placement
- Cache management
- State serialization

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Documentation

- [Model Card](MODEL_CARD.md): Detailed model specifications
- [Data Sources](DATA_SOURCES.md): Training data information
- [Examples](examples/): Usage examples and notebooks
- [API Reference](docs/api.md): API documentation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@software{promethean_context_engine,
  title={Promethean Context Engine: Memory-Efficient LLM Implementation},
  author={Luiz Frias},
  year={2025},
  url={https://github.com/Luiz-Frias/LLM-Dev}
}
```

## Acknowledgements

Special thanks to:
- The Mamba team for inspiration on selective state space models
- The Hugging Face team for transformers library
- The PyTorch team for MPS support 