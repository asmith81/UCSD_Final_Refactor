#!/bin/bash
# Setup script for configuring environment on RunPod or local machine
# This script validates dependencies, creates necessary directories,
# and sets up environment variables needed for the project.

set -e  # Exit immediately if a command exits with a non-zero status

# Detect environment
if [[ -n "$RUNPOD_POD_ID" ]]; then
    ENVIRONMENT="runpod"
    echo "📋 Detected RunPod environment"
else
    ENVIRONMENT="local"
    echo "📋 Detected local environment"
fi

# Find project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
echo "📂 Project root: $PROJECT_ROOT"

# Export project root as environment variable
export PROJECT_ROOT

# Create logs directory
mkdir -p "$PROJECT_ROOT/logs"

# Check Python version
PYTHON_VERSION=$(python3 --version)
echo "🐍 $PYTHON_VERSION"

# Check for GPU if on RunPod
if [[ "$ENVIRONMENT" == "runpod" ]]; then
    if command -v nvidia-smi &> /dev/null; then
        echo "🔍 Checking GPU..."
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader)
        echo "🖥️ GPU: $GPU_INFO"
    else
        echo "⚠️ Warning: NVIDIA tools not found, but running in RunPod environment"
    fi
fi

# Install or verify required packages
echo "📦 Checking required packages..."

# Create a temporary requirements file
TMP_REQUIREMENTS=$(mktemp)

if [[ "$ENVIRONMENT" == "runpod" ]]; then
    # RunPod-specific requirements
    cat > "$TMP_REQUIREMENTS" << EOF
transformers>=4.34.0
torch>=2.0.0
accelerate>=0.21.0
sentencepiece>=0.1.99
tqdm>=4.66.1
pillow>=10.0.0
matplotlib>=3.7.2
pandas>=2.0.3
python-Levenshtein>=0.21.1
bitsandbytes>=0.41.1
numpy>=1.24.3
pyyaml>=6.0
EOF
else
    # Local environment requirements (can be lighter)
    cat > "$TMP_REQUIREMENTS" << EOF
transformers>=4.34.0
torch>=2.0.0
pillow>=10.0.0
matplotlib>=3.7.2
pandas>=2.0.3
pyyaml>=6.0
EOF
fi

# Install packages
pip install -r "$TMP_REQUIREMENTS"
rm "$TMP_REQUIREMENTS"

# Verify installation
echo "✅ Validating installations..."
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
python3 -c "import transformers; print(f'Transformers {transformers.__version__}')"
python3 -c "import yaml; print(f'PyYAML {yaml.__version__}')"

# Create cache directory for models
if [[ "$ENVIRONMENT" == "runpod" ]]; then
    # On RunPod, use a persistent directory
    if [[ -d "/cache" ]]; then
        mkdir -p "/cache/models"
        echo "📂 Using RunPod persistent cache: /cache/models"
    else
        mkdir -p "$PROJECT_ROOT/models/cache"
        echo "📂 Created model cache directory: $PROJECT_ROOT/models/cache"
    fi
else
    # Local environment
    mkdir -p "$PROJECT_ROOT/models/cache"
    echo "📂 Created model cache directory: $PROJECT_ROOT/models/cache"
fi

# Create necessary local directories
mkdir -p "$PROJECT_ROOT/data/images"
mkdir -p "$PROJECT_ROOT/results/raw"
mkdir -p "$PROJECT_ROOT/results/processed"

# Set up Python path to include project
export PYTHONPATH="${PYTHONPATH}:${PROJECT_ROOT}"
echo "🔄 Added $PROJECT_ROOT to PYTHONPATH"

# Additional environment-specific setup
if [[ "$ENVIRONMENT" == "runpod" ]]; then
    # RunPod-specific environment variables
    export HF_HOME="/cache/huggingface"  # Store HF models in persistent storage
    mkdir -p "$HF_HOME"
    echo "📂 Set Hugging Face cache to $HF_HOME"
    
    # Check CUDA version
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
        echo "🔧 CUDA Version: $CUDA_VERSION"
    fi
fi

# Run a Python validation script if it exists
VALIDATION_SCRIPT="$PROJECT_ROOT/src/config/validate_env.py"
if [[ -f "$VALIDATION_SCRIPT" ]]; then
    echo "🔍 Running environment validation..."
    python3 "$VALIDATION_SCRIPT"
fi

echo "✨ Environment setup complete for $ENVIRONMENT!"
echo "📋 To use this configuration in your notebooks, import the environment modules:"
echo "    from src.config.environment import get_environment_config"
echo "    from src.config.paths import get_path_config"
echo "    from src.config.experiment import get_experiment_config"
echo ""
echo "🚀 Ready to start experiments!"

# Make the script executable
chmod +x "$VALIDATION_SCRIPT" 2>/dev/null || true

# Final instructions for RunPod
if [[ "$ENVIRONMENT" == "runpod" ]]; then
    echo ""
    echo "💡 RunPod Tips:"
    echo "  - Use '/cache' for persistent storage between sessions"
    echo "  - Remember to save important results outside the container"
    echo "  - Use 'nvidia-smi' to monitor GPU usage during experiments"
fi