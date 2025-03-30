# Invoice Extraction System: Architecture Overview

This document provides a high-level overview of the invoice extraction system architecture, component relationships, and data flow.

## System Architecture Diagram

```
+--------------------------------------------------------------------------------------------------+
|                                   Invoice Extraction System                                      |
+--------------------------------------------------------------------------------------------------+
                                              |
                    +------------------------+------------------------+
                    |                        |                        |
     +--------------v----------+  +---------v-----------+  +---------v-----------+
     |                         |  |                     |  |                     |
     | Configuration System    |  | Model Management    |  | Prompt System       |
     |                         |  |                     |  |                     |
     | - Environment Config    |  | - Model Service     |  | - Prompt Registry   |
     | - Path Config           |  | - Model Registry    |  | - Prompt Templates  |
     | - Experiment Config     |  | - Quantization      |  | - Field Prompts     |
     | - Prompt Config         |  | - Model Loading     |  | - Prompt Formatting |
     +--------------+----------+  +---------+-----------+  +---------+-----------+
                    |                        |                        |
                    +------------------------+------------------------+
                                              |
                                      +-------v--------+
                                      |                |
                                      | Execution      |
                                      | Pipeline       |
                                      |                |
                                      | - Factory      |
                                      | - Service      |
                                      | - Stages       |
                                      | - Recovery     |
                                      | - Error Handling|
                                      +-------+--------+
                                              |
                    +------------------------+------------------------+
                    |                        |                        |
     +--------------v----------+  +---------v-----------+  +---------v-----------+
     |                         |  |                     |  |                     |
     | Results Management      |  | Analysis            |  | Visualization       |
     |                         |  |                     |  |                     |
     | - Results Collector     |  | - Metrics           |  | - Visualizers       |
     | - Experiment Loader     |  | - Comparison        |  | - Dashboard         |
     | - Experiment Comparator |  | - Quantization      |  | - Interactive       |
     | - Storage               |  | - Statistical       |  | - Report Generation |
     +--------------------------+  +---------------------+  +---------------------+
```

## Core Components

### 1. Configuration System

The configuration system manages all settings and parameters for the extraction pipeline.

**Key Components:**
- **Base Config**: Abstract base class for all configuration objects
- **Environment Config**: Manages environment-specific settings (paths, resources)
- **Path Config**: Handles file and directory paths
- **Experiment Config**: Defines experiment parameters and settings
- **Prompt Config**: Manages prompt templates and parameters

**Responsibilities:**
- Load configuration from files, environment variables, and defaults
- Validate configuration values
- Provide typed configuration objects to other components
- Support runtime configuration overrides

### 2. Model Management

The model management system handles loading, caching, and interaction with LLM models.

**Key Components:**
- **Model Service**: Central service for model operations
- **Model Registry**: Maintains registry of available models
- **Quantization**: Implements model quantization strategies
- **Model Loading**: Efficient model loading with memory management

**Responsibilities:**
- Load models with appropriate settings
- Manage GPU memory efficiently
- Handle model quantization
- Provide unified interface for model interactions
- Track model performance metrics

### 3. Prompt System

The prompt system manages all prompts used in the extraction process.

**Key Components:**
- **Prompt Registry**: Central registry of all available prompts
- **Prompt Templates**: Parameterized prompt templates
- **Field Prompts**: Field-specific prompt collections
- **Prompt Formatting**: Format prompts for different models

**Responsibilities:**
- Manage prompt versions and categories
- Provide field-specific prompt collections
- Format prompts for specific models and contexts
- Support prompt experimentation

### 4. Execution Pipeline

The execution pipeline orchestrates the end-to-end extraction process.

**Key Components:**
- **Pipeline Factory**: Creates configured pipeline instances
- **Pipeline Service**: Coordinates overall execution
- **Pipeline Stages**: Individual processing stages
- **Recovery Mechanism**: Handles failure recovery
- **Error Handling**: Manages errors and exceptions

**Responsibilities:**
- Execute extraction workflows in stages
- Track progress and status
- Handle errors and recovery
- Collect and store results
- Manage resource utilization

### 5. Results Management

The results management system handles storage, retrieval, and analysis of experiment results.

**Key Components:**
- **Results Collector**: Collects and organizes results
- **Experiment Loader**: Loads and discovers experiment results
- **Experiment Comparator**: Compares multiple experiments
- **Storage**: Manages storage of experiment data

**Responsibilities:**
- Store extraction results in structured format
- Calculate performance metrics
- Enable experiment comparison and analysis
- Provide data for visualization

### 6. Analysis & Visualization

The analysis and visualization components provide insights into extraction performance.

**Key Components:**
- **Metrics**: Calculate performance metrics
- **Comparison**: Compare results across dimensions
- **Quantization Analysis**: Analyze quantization impact
- **Visualization**: Generate visualizations and reports

**Responsibilities:**
- Analyze extraction performance
- Generate visualizations
- Support interactive exploration
- Create comprehensive reports

## Data Flow

1. **Configuration** is loaded and validated
2. **Pipeline Factory** creates pipeline based on configuration
3. **Pipeline Service** orchestrates execution
4. **Model Service** loads and manages models
5. **Prompt System** provides formatted prompts
6. **Pipeline Stages** process data sequentially
7. **Results Collector** stores and analyzes results
8. **Visualization** generates insights from results

## Key Interfaces

### ModelService Interface

```python
class ModelService:
    def get_model(self, model_name: str, **kwargs) -> Any:
        """Get a model instance by name with optional parameters"""
        
    def generate(self, prompt: str, model_name: str, **kwargs) -> str:
        """Generate text using the specified model"""
        
    def get_embedding(self, text: str, model_name: str) -> List[float]:
        """Get embedding vector for the input text"""
```

### PipelineService Interface

```python
class PipelineService:
    def run_pipeline(self, config: ExperimentConfiguration) -> ExperimentResult:
        """Run the complete pipeline with the given configuration"""
        
    def create_checkpoint(self) -> str:
        """Create a checkpoint of the current pipeline state"""
        
    def resume_from_checkpoint(self, checkpoint_id: str) -> None:
        """Resume pipeline execution from a checkpoint"""
```

### ResultsCollector Interface

```python
class ResultsCollector:
    def add_extraction_result(self, result: ExtractionResult) -> None:
        """Add an extraction result to the collection"""
        
    def compare_experiments(self, experiment_ids: List[str]) -> ComparisonResult:
        """Compare multiple experiments"""
        
    def generate_experiment_dashboard(self) -> str:
        """Generate a comprehensive dashboard for the experiment"""
```

## Design Patterns

The system architecture implements several design patterns:

1. **Service Pattern**: For model management and pipeline execution
2. **Factory Pattern**: For creating pipeline and configuration objects
3. **Strategy Pattern**: For different extraction and quantization strategies
4. **Registry Pattern**: For models and prompts
5. **Pipeline Pattern**: For multi-stage processing
6. **Repository Pattern**: For results storage and retrieval

## Extensibility Points

The system is designed for extensibility at several points:

1. **New Models**: Register in model registry
2. **New Prompts**: Add to prompt registry
3. **New Pipeline Stages**: Implement the Stage interface
4. **New Metrics**: Add to metrics calculator
5. **New Visualization Types**: Extend visualizer classes
6. **New Experiment Types**: Add to experiment configuration 