from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
from ..models.service import ModelService
from ..prompts.service import PromptService
from .factory import PipelineFactory
from .service import ExtractionPipelineService
from ..config.experiment import ExperimentConfiguration
from ..results.collector import ResultsCollector
from ..analysis.advanced_metrics import AdvancedMetricsCalculator

class ExperimentRunner:
    """A debug-friendly interface for running invoice extraction experiments."""
    
    def __init__(self, workspace_path: Path):
        self.workspace_path = workspace_path
        self.model_service = None
        self.prompt_service = None
        self.pipeline_factory = None
        self.experiment_config = None
        self.setup_logging()
    
    def setup_logging(self):
        """Configure logging for the experiment runner."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('ExperimentRunner')
    
    def initialize_environment(self) -> bool:
        """Initialize the experiment environment."""
        try:
            # Validate workspace
            if not self.workspace_path.exists():
                raise ValueError(f"Workspace path does not exist: {self.workspace_path}")
            
            # Initialize services
            self.model_service = ModelService()
            self.prompt_service = PromptService()
            self.pipeline_factory = PipelineFactory()
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize environment: {str(e)}")
            return False
    
    def setup_system(self) -> bool:
        """Set up the system components."""
        try:
            # Initialize model registry
            self.model_service.initialize_registry()
            
            # Set up prompt management
            self.prompt_service.initialize()
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to setup system: {str(e)}")
            return False
    
    def configure_experiment(
        self,
        models: List[str],
        fields: List[str],
        prompts: Dict[str, str],
        quantization: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Configure the experiment parameters."""
        try:
            self.experiment_config = ExperimentConfiguration(
                models=models,
                fields=fields,
                prompts=prompts,
                quantization=quantization
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to configure experiment: {str(e)}")
            return False
    
    def execute_experiment(self) -> Dict[str, Any]:
        """Execute the configured experiment."""
        try:
            if not self.experiment_config:
                raise ValueError("Experiment not configured")
            
            # Create pipeline
            pipeline = self.pipeline_factory.create_pipeline(self.experiment_config)
            
            # Execute pipeline
            pipeline_service = ExtractionPipelineService(pipeline)
            results = pipeline_service.execute()
            
            return results
        except Exception as e:
            self.logger.error(f"Failed to execute experiment: {str(e)}")
            raise
    
    def analyze_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze and process experiment results using existing infrastructure."""
        try:
            # Use ResultsCollector for cross-field metrics
            collector = ResultsCollector(results)
            cross_field_metrics = collector.calculate_cross_field_metrics()
            
            # Use AdvancedMetricsCalculator for detailed analysis
            calculator = AdvancedMetricsCalculator()
            performance_metrics = calculator.calculate_extraction_performance_metrics(results)
            
            return {
                'cross_field_metrics': cross_field_metrics,
                'performance_metrics': performance_metrics
            }
        except Exception as e:
            self.logger.error(f"Failed to analyze results: {str(e)}")
            return {} 