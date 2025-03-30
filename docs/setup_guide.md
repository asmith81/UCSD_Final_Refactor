# Invoice Extraction System: RunPod Setup Guide

This guide walks you through setting up the invoice extraction system on RunPod, from initial environment configuration to validation.

## Environment Requirements

### Recommended RunPod Configuration

| Resource | Minimum Specification | Recommended Specification |
|----------|----------------------|--------------------------|
| GPU      | NVIDIA T4 (16GB VRAM) | NVIDIA A10G (24GB VRAM) or better |
| CPU      | 4 vCPU cores | 8+ vCPU cores |
| RAM      | 16GB | 32GB+ |
| Storage  | 50GB | 100GB+ |
| Image    | PyTorch 2.0+ with CUDA 11.8+ | Latest PyTorch with CUDA support |

### RunPod Template Selection

1. From the RunPod dashboard, select "Deploy" 
2. Choose "PyTorch" under the template section
3. Select a GPU that meets the minimum specifications above
4. Configure your pod with at least 50GB disk space
5. Deploy the pod

## Initial Setup

Once your RunPod is active, connect via HTTP (Jupyter Lab) or SSH (Terminal).

### System Setup (Run as Root)

Open a terminal and run the following commands:

```bash
# Update system packages
apt update
apt upgrade -y

# Install system dependencies
apt install -y build-essential git wget curl unzip libgl1 libglib2.0-0

# Install Python packages required for development
pip install --upgrade pip
pip install jupyterlab ipywidgets matplotlib
```

### Project Setup

1. Clone the repository:

```bash
cd /workspace
git clone https://github.com/your-organization/invoice-extraction.git
cd invoice-extraction
```

2. Install project dependencies:

```bash
pip install -r requirements.txt
```

## Environment Configuration

The system needs proper configuration to access resources and models.

### Setting Path Configuration

Create a `.env` file in the project root (or edit if it exists):

```bash
cat > .env << 'EOL'
# Base paths
DATA_DIR=/workspace/invoice-extraction/data
MODELS_DIR=/workspace/invoice-extraction/models
RESULTS_DIR=/workspace/invoice-extraction/results
CONFIGS_DIR=/workspace/invoice-extraction/configs

# RunPod specific settings
USE_GPU=True
CUDA_VISIBLE_DEVICES=0
MAX_MEMORY=auto
EOL
```

### Model Downloads

1. Create required directories:

```bash
mkdir -p models/llm
mkdir -p data/invoices
mkdir -p results
```

2. Download models (if not already available):

```bash
# Example for downloading a model (modify as needed)
python scripts/download_models.py --model llm/llama-7b
```

## Validation Steps

Run these validation steps to ensure your environment is properly configured:

### 1. Environment Check

Run the validation script to check hardware, dependencies, and configuration:

```bash
python -m src.config.validate_env
```

### 2. GPU Verification

Verify GPU availability and memory:

```bash
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}'); print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB' if torch.cuda.is_available() else 'None')"
```

### 3. Model Loading Test

Test loading a model to verify CUDA compatibility:

```bash
python scripts/test_model_loading.py
```

## Running Your First Experiment

Once your environment is validated, you're ready to run your first experiment.

1. Start Jupyter Lab (if not already running):

```bash
jupyter lab --ip=0.0.0.0 --port=8888 --allow-root
```

2. Navigate to the `notebooks` directory and open `01_environment_setup.ipynb`

3. Run the notebook cells to perform a complete environment check

4. Then proceed to `02_experiment_configuration.ipynb` to configure and run your first extraction experiment

## Troubleshooting Common Issues

### CUDA Out of Memory

If you encounter CUDA out of memory errors:

1. Reduce batch size in experiment configuration
2. Use a quantized model variant
3. Try the memory optimization strategy:

```python
from src.config.experiment_config import create_experiment_config

config = create_experiment_config(
    experiment_type="BASIC_EXTRACTION",
    model_name="llama-7b",
    memory_optimization=True,
    batch_size=1  # Reduce batch size
)
```

### Model Loading Failures

If models fail to load:

1. Check GPU memory availability
2. Verify model paths in environment configuration
3. Try quantization:

```python
from src.config.experiment_config import create_experiment_config

config = create_experiment_config(
    experiment_type="BASIC_EXTRACTION",
    model_name="llama-7b",
    quantization={"strategy": "4bit", "use_double_quant": True}
)
```

### Network Connection Issues

If you encounter issues downloading models:

1. Verify RunPod has internet connectivity
2. Try downloading to local machine and uploading to RunPod via SCP
3. Use shared volumes to access pre-downloaded models

## Next Steps

After successful setup:
1. Proceed to the experiment configuration notebook to start extraction tasks
2. Review the architecture documentation in `docs/architecture.md` for system understanding
3. Explore the interface documentation in `docs/interfaces.md` for component details

For additional assistance, refer to project documentation or contact support. 