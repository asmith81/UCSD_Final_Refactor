# Invoice Extraction System: Key Interfaces

This document outlines the key interfaces within the invoice extraction system, focusing on how components interact with each other. Understanding these interfaces is essential for extending the system or implementing custom components.

## Core Interface Overview

| Interface | Purpose | Key Implementation |
|-----------|---------|-------------------|
| `ExperimentConfiguration` | Define experiment parameters | `src/config/experiment_config.py` |
| `ModelService` | Interface for model operations | `src/models/model_service.py` |
| `PipelineStage` | Interface for pipeline stages | `src/execution/pipeline/base.py` |
| `PipelineService` | Control pipeline execution | `src/execution/pipeline/service.py` |
| `PipelineFactory` | Create pipeline instances | `src/execution/pipeline/factory.py` |
| `ProgressTracker` | Track pipeline progress | `src/execution/pipeline/error_handling.py` |
| `ResultsCollector` | Manage experiment results | `src/results/collector.py` |
| `ExperimentLoader` | Load experiment results | `src/results/collector.py` |

## Configuration Interfaces

### ExperimentConfiguration

The central configuration interface that defines parameters for extraction experiments.

```python
@dataclass
class ExperimentConfiguration:
    # Metadata
    experiment_name: str
    experiment_type: ExperimentType
    description: Optional[str] = None
    
    # Core extraction parameters
    model_name: str
    fields: List[str]
    batch_size: int = 1
    
    # Prompt configuration
    prompts: Dict[str, str] = field(default_factory=dict)
    prompt_template: Optional[str] = None
    
    # Quantization strategy
    quantization: Optional[Dict[str, Any]] = None
    
    # Processing options
    memory_optimization: bool = False
    use_checkpoint: bool = False
    
    # Result validation
    validate_results: bool = True
    validation_method: str = "exact_match"
```

#### Key Methods

```python
def create_experiment_config(
    experiment_type: Union[str, ExperimentType],
    model_name: str,
    fields: Optional[List[str]] = None,
    prompts: Optional[Dict[str, str]] = None,
    **kwargs
) -> ExperimentConfiguration:
    """Create an experiment configuration with the given parameters."""
```

```python
def load_experiment_config(config_path: Union[str, Path]) -> ExperimentConfiguration:
    """Load an experiment configuration from a file."""
```

### ExperimentTemplate

Templates for common experiment configurations.

```python
@dataclass
class ExperimentTemplate:
    name: str
    template_type: str
    description: str
    base_config: Dict[str, Any]
    
    def create_configuration(self, **overrides) -> ExperimentConfiguration:
        """Create an experiment configuration from this template."""
```

## Model Service Interface

The interface for interacting with LLM models.

```python
class ModelService:
    def get_model(
        self, 
        model_name: str, 
        quantization: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Any:
        """Get a model instance by name with specified parameters."""
        
    def generate(
        self, 
        prompt: str, 
        model_name: str, 
        image: Optional[Any] = None, 
        **kwargs
    ) -> str:
        """Generate text using the specified model."""
        
    def get_embedding(
        self, 
        text: str, 
        model_name: str = "default"
    ) -> List[float]:
        """Get an embedding vector for the input text."""
        
    def unload_model(self, model_name: str) -> bool:
        """Unload a model from memory."""
        
    def get_available_models(self) -> List[str]:
        """Get a list of available model names."""
```

## Pipeline Interfaces

### PipelineStage

The interface for individual pipeline processing stages.

```python
class PipelineStage(ABC):
    @abstractmethod
    def process(
        self, 
        data: Any, 
        context: Dict[str, Any]
    ) -> Tuple[Any, Dict[str, Any]]:
        """Process data and update context."""
        
    @abstractmethod
    def get_stage_name(self) -> str:
        """Get the name of this stage."""
        
    @property
    def requires_model(self) -> bool:
        """Whether this stage requires a model."""
        
    def cleanup(self) -> None:
        """Clean up resources used by this stage."""
```

### PipelineService

The interface for controlling pipeline execution.

```python
class PipelineService:
    def run_pipeline(
        self, 
        config: ExperimentConfiguration
    ) -> ExperimentResult:
        """Run the complete pipeline with the given configuration."""
        
    def run_with_data(
        self, 
        data: Any, 
        config: ExperimentConfiguration
    ) -> ExperimentResult:
        """Run the pipeline with the given data and configuration."""
        
    def create_checkpoint(self) -> str:
        """Create a checkpoint of the current pipeline state."""
        
    def resume_from_checkpoint(
        self, 
        checkpoint_id: str
    ) -> ExperimentResult:
        """Resume pipeline execution from a checkpoint."""
        
    def get_progress(self) -> Dict[str, Any]:
        """Get the current progress of the pipeline."""
        
    def stop_pipeline(self) -> None:
        """Stop the pipeline execution."""
```

### PipelineFactory

The interface for creating pipeline instances.

```python
class PipelineFactory:
    @classmethod
    def create_pipeline(
        cls, 
        config: ExperimentConfiguration
    ) -> Tuple[List[PipelineStage], PipelineService]:
        """Create a pipeline for the given configuration."""
        
    @classmethod
    def create_model_comparison_pipeline(
        cls,
        models: List[str],
        fields: List[str],
        prompts: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Tuple[ExperimentConfiguration, PipelineService]:
        """Create a pipeline for comparing models."""
        
    @classmethod
    def create_prompt_comparison_pipeline(
        cls,
        model_name: str,
        fields: List[str],
        prompt_variants: Dict[str, Dict[str, str]],
        **kwargs
    ) -> Tuple[ExperimentConfiguration, PipelineService]:
        """Create a pipeline for comparing prompts."""
        
    @classmethod
    def create_quantization_pipeline(
        cls,
        model_name: str,
        fields: List[str],
        quantization_strategies: List[Dict[str, Any]],
        **kwargs
    ) -> Tuple[ExperimentConfiguration, PipelineService]:
        """Create a pipeline for comparing quantization strategies."""
        
    @classmethod
    def create_notebook_pipeline(
        cls,
        experiment_type: str,
        model_name: str,
        fields: List[str],
        **kwargs
    ) -> Tuple[ExperimentConfiguration, PipelineService]:
        """Create a notebook-friendly pipeline."""
```

### ProgressTracker

The interface for tracking pipeline progress.

```python
class ProgressTracker:
    def start_stage(self, stage_name: str, total_items: int) -> None:
        """Start tracking a new stage."""
        
    def update_progress(
        self, 
        items_completed: int, 
        status: str = "processing"
    ) -> None:
        """Update the progress of the current stage."""
        
    def complete_stage(self) -> None:
        """Mark the current stage as complete."""
        
    def mark_stage_failed(
        self, 
        error_message: str, 
        error_type: str = "processing_error"
    ) -> None:
        """Mark the current stage as failed."""
        
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get a summary of the progress."""
        
    def get_estimated_time_remaining(self) -> float:
        """Get the estimated time remaining for the pipeline."""
```

## Results Management Interfaces

### ResultsCollector

The interface for collecting and analyzing results.

```python
class ResultsCollector:
    def add_extraction_result(
        self, 
        result: ExtractionResult, 
        field_name: str, 
        prompt_name: str
    ) -> None:
        """Add an extraction result to the collection."""
        
    def save_results(self, path: Optional[Union[str, Path]] = None) -> str:
        """Save the current results to disk."""
        
    def load_results(self, path: Union[str, Path]) -> None:
        """Load results from disk."""
        
    def generate_experiment_result(self) -> ExperimentResult:
        """Generate a complete experiment result."""
        
    def compare_fields(self, metrics: Optional[List[str]] = None) -> ComparisonResult:
        """Compare results across different fields."""
        
    def compare_prompts(
        self, 
        field: str, 
        metrics: Optional[List[str]] = None
    ) -> ComparisonResult:
        """Compare results across different prompts for a field."""
        
    def find_similar_experiments(
        self,
        model_name: Optional[str] = None,
        field: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Find experiments similar to the current one."""
        
    def compare_with_experiment(
        self,
        other_experiment_id: str,
        field: Optional[str] = None,
        metrics: Optional[List[str]] = None
    ) -> ComparisonResult:
        """Compare this experiment with another experiment."""
        
    def generate_experiment_dashboard(
        self,
        output_path: Optional[Union[str, Path]] = None,
        format: str = "html"
    ) -> str:
        """Generate a comprehensive dashboard for this experiment."""
```

### ExperimentLoader

The interface for loading and discovering experiment results.

```python
class ExperimentLoader:
    def discover_experiments(
        self,
        force_refresh: bool = False,
        include_subdirectories: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """Discover available experiment results."""
        
    def load_experiment(
        self,
        experiment_id_or_path: Union[str, Path]
    ) -> Optional[Tuple[ExperimentResult, Any]]:
        """Load a complete experiment by ID or path."""
        
    def filter_experiments(
        self,
        model: Optional[str] = None,
        fields: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """Filter discovered experiments based on criteria."""
        
    def get_experiment_dataframe(self) -> "pd.DataFrame":
        """Convert discovered experiments to a DataFrame."""
        
    def batch_load_experiments(
        self,
        experiment_ids: List[str],
        batch_size: int = 10
    ) -> Dict[str, Any]:
        """Load multiple experiments in batch mode."""
```

### ExperimentComparator

The interface for comparing experiment results.

```python
class ExperimentComparator:
    def compare_experiments(
        self,
        experiment_ids: List[str],
        field: Optional[str] = None,
        prompt_name: Optional[str] = None,
        metrics: Optional[List[str]] = None,
        comparison_dimension: str = "experiment"
    ) -> ComparisonResult:
        """Compare multiple experiments based on specific parameters."""
        
    def compare_models(
        self,
        model_names: List[str],
        field: Optional[str] = None,
        prompt_name: Optional[str] = None,
        metrics: Optional[List[str]] = None
    ) -> ComparisonResult:
        """Compare performance across different models."""
        
    def compare_quantization_strategies(
        self,
        model_name: str,
        strategies: List[str],
        field: Optional[str] = None,
        metrics: Optional[List[str]] = None
    ) -> ComparisonResult:
        """Compare performance across different quantization strategies."""
```

## Notebook Support Interfaces

### ExperimentUtils

Utilities for configuring and running experiments in notebooks.

```python
def create_basic_experiment(
    model_name: str,
    fields: List[str],
    memory_optimization: bool = False
) -> ExperimentConfiguration:
    """Create a basic experiment configuration."""
    
def run_extraction_experiment(
    config: ExperimentConfiguration,
    data_path: Union[str, Path]
) -> ExperimentResult:
    """Run an extraction experiment with the given configuration."""
    
def load_experiment_template(
    template_name: str,
    **kwargs
) -> ExperimentConfiguration:
    """Load a predefined experiment template."""
    
def list_available_templates() -> List[str]:
    """List all available experiment templates."""
    
def visualize_experiment_results(
    result: ExperimentResult,
    output_format: str = "notebook"
) -> Any:
    """Visualize experiment results in the notebook."""
```

### SetupUtils

Utilities for environment setup and validation in notebooks.

```python
def validate_environment() -> Dict[str, bool]:
    """Validate the runtime environment."""
    
def check_gpu_availability() -> Dict[str, Any]:
    """Check GPU availability and memory."""
    
def configure_paths(
    data_dir: Optional[str] = None,
    models_dir: Optional[str] = None,
    results_dir: Optional[str] = None
) -> Dict[str, str]:
    """Configure system paths."""
    
def install_dependencies(
    requirements: Optional[List[str]] = None
) -> bool:
    """Install required dependencies."""
    
def test_model_loading(
    model_name: str
) -> Dict[str, Any]:
    """Test loading a model to verify setup."""
```

## Using These Interfaces

### Example: Creating and Running an Experiment

```python
from src.config.experiment_config import create_experiment_config
from src.execution.pipeline.factory import PipelineFactory

# Create experiment configuration
config = create_experiment_config(
    experiment_type="BASIC_EXTRACTION",
    model_name="llama-7b",
    fields=["invoice_number", "invoice_date", "total_amount"],
    quantization={"strategy": "4bit"}
)

# Create pipeline
_, pipeline_service = PipelineFactory.create_pipeline(config)

# Run pipeline
result = pipeline_service.run_pipeline(config)

# Save results
from src.results.collector import ResultsCollector
collector = ResultsCollector(base_path="results", experiment_name=config.experiment_name)
collector.save_results()

# Generate dashboard
dashboard_path = collector.generate_experiment_dashboard()
```

### Example: Comparing Experiments

```python
from src.results.collector import ExperimentLoader, ExperimentComparator

# Load experiment results
loader = ExperimentLoader(base_directory="results")
experiments = loader.filter_experiments(
    model="llama-7b",
    fields=["invoice_number"]
)

# Compare experiments
comparator = ExperimentComparator(loader)
comparison = comparator.compare_experiments(
    experiment_ids=list(experiments.keys())[:3],
    field="invoice_number",
    metrics=["accuracy", "processing_time"]
)

# Visualize comparison
from src.analysis.visualization import create_comparison_chart
create_comparison_chart(comparison, chart_type="bar") 