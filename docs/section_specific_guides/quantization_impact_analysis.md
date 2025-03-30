# Quantization Impact Analysis

This document provides an overview of the quantization impact analysis functionality 
implemented in the `src/analysis/quantization.py` module.

## Overview

The quantization impact analysis module enables comprehensive evaluation of different 
quantization strategies for vision-language models, focusing on understanding the 
trade-offs between accuracy, memory usage, and inference speed. This analysis is
crucial for deploying models in resource-constrained environments.

## Key Components

### 1. Hardware Profiling

The `HardwareProfile` class detects and profiles the current hardware environment:

- GPU availability and memory capacity
- CUDA compute capability
- CPU memory availability
- Compatibility with different quantization strategies

### 2. Memory Tracking

Enhanced memory tracking during model loading and inference:

- Detailed GPU memory usage monitoring
- Per-device memory statistics
- Memory timeline tracking throughout the model lifecycle
- Fragmentation analysis

### 3. Efficiency Metrics

Comprehensive metrics for evaluating quantization efficiency:

- Memory efficiency (accuracy per GB)
- Speed-accuracy trade-offs
- Memory savings compared to baseline
- Inference overhead analysis

### 4. Benchmarking

The `QuantizationAnalyzer` class provides methods for benchmarking different 
quantization strategies:

- Automated testing of multiple quantization approaches
- Comparative performance analysis
- Optimal strategy selection based on hardware constraints
- Recommendation generation

## Using the Quantization Analyzer

```python
from src.analysis.quantization import get_quantization_analyzer

# Create an analyzer for a specific model
analyzer = get_quantization_analyzer(
    model_name="openai/clip-vit-large-patch14",
    config=experiment_config,
    results_dir="./results"
)

# Run a comprehensive benchmark
benchmark = analyzer.run_benchmark(
    strategies=["none", "int8", "int4", "gptq-int4", "awq-int4"], 
    priority="balanced"
)

# Access results
for result in benchmark.results:
    print(f"Strategy: {result.strategy}")
    print(f"  Memory: {result.memory_used_gb:.2f} GB")
    print(f"  Accuracy: {result.accuracy:.4f}")
    print(f"  Inference time: {result.inference_time:.4f} s")
    print(f"  Efficiency score: {result.efficiency_score:.4f}")

# Get recommendations
print("\nRecommendations:")
for recommendation in benchmark.recommendations:
    print(f"- {recommendation}")
```

## Memory Analysis

The enhanced memory analysis provides detailed insights into:

1. **Total Memory Usage**: Breakdown of memory consumption during model loading and inference
2. **Memory Savings**: Comparison of memory savings relative to a baseline (unquantized model)
3. **Fragmentation Analysis**: Detection of memory fragmentation issues during model lifecycle
4. **Memory Efficiency**: Evaluation of accuracy per GB of memory used
5. **Timeline Analysis**: Tracking of memory usage over time during benchmarking
6. **Overhead Analysis**: Measurement of additional memory required during inference

## Visualization Data

The module generates structured data suitable for visualization:

- Memory usage charts
- Speed-accuracy trade-off plots
- Radar charts for comparing strategies
- Memory timeline visualizations
- Normalized comparisons for side-by-side evaluation

## Hardware-Adaptive Recommendations

The system provides intelligent recommendations based on:

- Current hardware constraints
- Specific priorities (accuracy, speed, memory)
- Detected memory fragmentation issues
- Optimal strategy for the current environment

## Integration with Results Management

The quantization analysis is integrated with the results management system:

- Benchmark results are automatically saved
- Results can be loaded and compared across experiments
- Visualization data is exportable to various formats

## Example: Memory Timeline Analysis

```python
# Get memory timeline data for visualization
visualization_data = analyzer._generate_visualization_data(benchmark)
memory_timeline = visualization_data["memory_timeline"]

# Plot memory usage over time for each strategy
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
for strategy, data in memory_timeline.items():
    plt.plot(data["timestamps"], data["memory_values"], label=strategy)

plt.xlabel("Time (s)")
plt.ylabel("Memory Usage (GB)")
plt.title("Memory Usage Timeline Across Quantization Strategies")
plt.legend()
plt.grid(True)
plt.show()
```

## Conclusion

The enhanced quantization impact analysis module provides a comprehensive framework
for evaluating and selecting optimal quantization strategies based on specific 
hardware constraints and performance requirements. This enables users to make
informed decisions when deploying models in resource-constrained environments. 