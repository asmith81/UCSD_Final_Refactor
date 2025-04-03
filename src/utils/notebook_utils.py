from pathlib import Path
import sys
import subprocess
import logging
from typing import Dict, List, Optional, Any
from ..execution.experiment_runner import ExperimentRunner

def setup_environment(requirements_path: Path) -> bool:
    """Install and validate dependencies."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_path)])
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to install dependencies: {str(e)}")
        return False

def configure_paths(workspace_path: Path) -> Dict[str, Path]:
    """Set up workspace paths."""
    paths = {
        'workspace': workspace_path,
        'data': workspace_path / 'data',
        'models': workspace_path / 'models',
        'results': workspace_path / 'results',
        'logs': workspace_path / 'logs'
    }
    
    # Create directories if they don't exist
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    
    return paths

def initialize_system(workspace_path: Path) -> ExperimentRunner:
    """Initialize the system and return an ExperimentRunner instance."""
    runner = ExperimentRunner(workspace_path)
    
    if not runner.initialize_environment():
        raise RuntimeError("Failed to initialize environment")
    
    if not runner.setup_system():
        raise RuntimeError("Failed to setup system")
    
    return runner

def create_experiment_config(
    models: List[str],
    fields: List[str],
    prompts: Dict[str, str],
    quantization: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create experiment configuration dictionary."""
    return {
        'models': models,
        'fields': fields,
        'prompts': prompts,
        'quantization': quantization or {}
    }

def run_experiment(
    runner: ExperimentRunner,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Execute the experiment and return results."""
    if not runner.configure_experiment(**config):
        raise RuntimeError("Failed to configure experiment")
    
    results = runner.execute_experiment()
    analysis = runner.analyze_results(results)
    
    return {
        'raw_results': results,
        'analysis': analysis
    }

def analyze_and_visualize(results: Dict[str, Any]) -> None:
    """Process and display experiment results."""
    # Basic visualization of results
    analysis = results['analysis']
    
    print("\nExperiment Results:")
    print(f"Total Models: {analysis['total_models']}")
    print(f"Total Fields: {analysis['total_fields']}")
    print(f"Success Rate: {analysis['success_rate']:.2%}")
    
    print("\nPerformance Metrics:")
    for metric, value in analysis['performance_metrics'].items():
        print(f"{metric}: {value}") 