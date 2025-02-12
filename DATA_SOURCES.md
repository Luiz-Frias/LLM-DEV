# Data Sources and Training Curriculum for Promethean Context Engine

## Primary Dataset

### C4 (Colossal Clean Crawled Corpus)
- **Source**: [C4 Dataset](https://huggingface.co/datasets/c4)
- **Language**: English
- **Size**: Configurable subset based on memory constraints
- **Quality**: High-quality filtered web text
- **Processing**: 
  - Adaptive cleaning and normalization
  - Predictive quality filtering
  - Maximum sequence length: 512 tokens
  - Dynamic batch size (max 2 for 8GB RAM)

## Adaptive Learning Phases

### Phase 1: Foundational Understanding (C4)
- **Dataset**: C4 (en subset)
- **Purpose**: Build foundational knowledge base
- **Features**:
  - High-quality web text
  - Diverse topics and styles
  - Natural language patterns
  - Real-world knowledge acquisition

### Phase 2: Contextual Reasoning (CoT)
- **Datasets**: 
  - GSM8K (mathematical reasoning)
  - StrategyQA (strategic thinking)
- **Purpose**: Enhance structured reasoning capabilities
- **Features**:
  - Step-by-step problem decomposition
  - Explicit reasoning chains
  - Complex problem solving
  - Contextual understanding

## Resource-Aware Data Loading

### Streaming Implementation
- Predictive batch sampling
- Memory-mapped caching
- Efficient collation
- Automatic resource management

### Batch Processing
- Hard cap at batch size 2
- Gradient accumulation (4 steps)
- Effective batch size: 8
- Memory-aware adjustments

## Data Quality Control

### Filtering Criteria
- Minimum sequence length: 10 tokens
- Maximum sequence length: 512 tokens
- Content quality metrics
- Language validation

### Processing Pipeline
1. Adaptive text cleaning
2. Predictive quality filtering
3. Length validation
4. Efficient tokenization
5. Dynamic batching

## Memory Management

### Hierarchical Memory Tree
- Predictive splitting based on capacity
- Efficient O(log n) operations
- Adaptive load balancing
- Automatic rebalancing

### Selective State Space Model (SSSM)
- State space parameter optimization
- Selective updates
- Spike thresholding
- Continuous-time dynamics

### Monolith Intake System
- Predictive batch processing
- Salience calculation
- Type validation
- Memory optimization

## Training Phases

### Initial Training (C4)
- **Duration**: 0-85% of total epochs
- **Synthetic Ratio**: 20%
- **Batch Size**: 2
- **Learning Rate**: 2e-5 to 5e-5

### Contextual Fine-tuning
- **Duration**: 85-100% of total epochs
- **Synthetic Ratio**: Up to 90%
- **Batch Size**: 2
- **Learning Rate**: 1e-5 to 2e-5

## Data Caching Strategy

### Disk Cache
- Dataset chunks
- Memory-mapped files
- Efficient streaming
- Automatic cleanup

### Memory Cache
- Recent examples
- High-salience items
- Frequently accessed patterns
- Dynamic size adjustment

## Performance Metrics

### Training
- Examples per second: ~10-20
- Memory usage: 6-7GB peak
- Disk cache: ~1GB
- GPU utilization: 80-90%

### Inference
- Latency: 50-100ms
- Memory usage: 2-3GB
- Throughput: 10-20 req/s
- Context window: 512 tokens

## License Compliance

### Attribution Requirements
- Wikipedia content maintains CC BY-SA 3.0 attribution
- Project Gutenberg maintains original author attribution
- Common Crawl follows terms of use guidelines

### Usage Restrictions
1. **Wikipedia**
   - Share-alike requirement for derivatives
   - Attribution required
   - Commercial use permitted

2. **Project Gutenberg**
   - Public domain in the US
   - Usage restrictions may apply in other countries
   - Attribution recommended but not required

3. **Common Crawl**
   - No redistribution restrictions
   - Attribution recommended
   - Commercial use permitted

## Quality Assurance

### Validation Process
1. License verification for each source
2. Content quality assessment
3. Bias detection and mitigation
4. Regular dataset audits

### Documentation Updates
This document is updated when:
- New data sources are added
- Processing methodology changes
- License terms are updated
- Quality issues are discovered

## Contact
For questions about data sources or processing methodology, please use the GitHub Issues system.

Last Updated: February 2025 