# Notebooks

This directory contains Jupyter notebooks for the invoice extraction system.

## Notebook Structure

1. `02_experiment_configuration.py`: Configures and runs extraction experiments
   - Sets up environment and paths
   - Validates GPU and dependencies
   - Configures and runs extraction experiments
2. `03_results_analysis.py`: Analyzes experiment results

## Important Notes

### Environment Setup
Each notebook runs in its own kernel (in environments like RunPod, SageMaker, etc.). This means:

1. Environment variables and paths set in one notebook's kernel are not available in another notebook's kernel
2. Each notebook needs to be self-contained and handle its own setup
3. The experiment configuration notebook includes all necessary environment setup code

For future improvements, consider:
- Using a Docker container to standardize the environment
- Setting up environment variables at the system level
- Creating a shared configuration module

## Usage

1. Use `02_experiment_configuration.py` to:
   - Set up your environment
   - Configure and run extraction experiments
2. Analyze results with `03_results_analysis.py` 