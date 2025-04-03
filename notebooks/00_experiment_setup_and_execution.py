# Invoice Extraction System
# Single Notebook Solution

## 1. Environment Setup
from pathlib import Path
import sys
from src.utils.notebook_utils import setup_environment, configure_paths

# Set workspace path
workspace_path = Path.cwd().parent
sys.path.append(str(workspace_path))

# Install dependencies
requirements_path = workspace_path / 'requirements.txt'
if not setup_environment(requirements_path):
    raise RuntimeError("Failed to setup environment")

# Configure paths
paths = configure_paths(workspace_path)

## 2. System Initialization
from src.utils.notebook_utils import initialize_system

# Initialize system
runner = initialize_system(workspace_path)

## 3. Experiment Configuration
from src.utils.notebook_utils import create_experiment_config

# Define experiment parameters
models = ["model1", "model2"]  # Replace with actual model names
fields = ["field1", "field2"]  # Replace with actual field names
prompts = {
    "field1": "Extract {field1} from the invoice",
    "field2": "Extract {field2} from the invoice"
}
quantization = {
    "bits": 4,
    "group_size": 128
}

# Create experiment configuration
experiment_config = create_experiment_config(
    models=models,
    fields=fields,
    prompts=prompts,
    quantization=quantization
)

## 4. Experiment Execution
from src.utils.notebook_utils import run_experiment

# Run experiment
results = run_experiment(runner, experiment_config)

## 5. Analysis and Visualization
from src.utils.notebook_utils import analyze_and_visualize

# Analyze and visualize results
analyze_and_visualize(results) 