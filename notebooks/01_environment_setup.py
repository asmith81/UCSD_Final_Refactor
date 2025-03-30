# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Invoice Extraction System: Environment Setup
#
# This notebook guides you through the process of setting up and validating your environment for invoice extraction experiments. By the end of this notebook, you'll have:
#
# 1. A properly configured environment with all necessary dependencies
# 2. Validated GPU setup with appropriate memory settings
# 3. Configured paths for data, models, and results
# 4. Verified model availability and compatibility
# 5. Tested the system with a simple extraction task
#
# Let's get started!

# %% [markdown]
# ## 1. Environment Detection
#
# First, let's detect your current environment and display key system information.

# %%
import os
import sys
import platform
from datetime import datetime

# Add the project root to the path if not already there
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import our utility modules
try:
    from src.notebook.setup_utils import (
        validate_environment, 
        check_gpu_availability, 
        configure_paths, 
        get_system_info
    )
    from src.notebook.error_utils import display_error, NotebookFriendlyError, safe_execute
    setup_utils_available = True
except ImportError as e:
    print(f"⚠️ Error importing setup utilities: {str(e)}")
    print("⚠️ Will use basic environment detection instead.")
    setup_utils_available = False

# Basic environment information
print(f"📋 Environment Setup and Validation")
print(f"🕒 Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"💻 System: {platform.system()} {platform.release()}")
print(f"🐍 Python version: {platform.python_version()}")
print(f"📂 Working directory: {os.getcwd()}")

# Check if we're in a notebook environment
try:
    from IPython import get_ipython
    in_notebook = get_ipython().__class__.__name__ == 'ZMQInteractiveShell'
    print(f"📓 Running in notebook environment: {in_notebook}")
except (ImportError, NameError, AttributeError):
    in_notebook = False
    print(f"📓 Running in script environment")

# Use our utility module if available
if setup_utils_available:
    system_info = get_system_info()
    print(f"\n📊 Detailed System Information:")
    print(f"🔍 Platform: {system_info['platform']}")
    print(f"🧠 Python implementation: {system_info['python_implementation']}")
    print(f"📂 Python path: {system_info['python_path']}")
    
    # Check if we're in a RunPod environment
    if system_info['in_runpod']:
        print(f"☁️ RunPod environment detected!")
        print(f"🆔 Pod ID: {system_info['env_vars'].get('RUNPOD_POD_ID', 'Unknown')}")
        print(f"💾 Disk space: {system_info['disk_space']['free_gb']:.1f} GB free of {system_info['disk_space']['total_gb']:.1f} GB")
    else:
        print(f"💻 Local environment detected")
else:
    # Fallback if our utility module isn't available
    def is_runpod():
        """Check if we're running in a RunPod environment"""
        return (
            os.environ.get("RUNPOD_POD_ID") is not None or 
            "A100" in os.environ.get("GPU_NAME", "") or
            "H100" in os.environ.get("GPU_NAME", "")
        )
    
    if is_runpod():
        print(f"☁️ RunPod environment detected!")
    else:
        print(f"💻 Local environment detected")

# %% [markdown]
# ## 2. GPU Configuration and Validation
#
# Now, let's check for GPU availability and configuration. This is crucial for efficient model inference.

# %%
import subprocess

# Use our utility module if available
if setup_utils_available:
    gpu_info = check_gpu_availability()
    
    if gpu_info['available']:
        print(f"✅ GPU detected: {gpu_info['name']}")
        print(f"📊 GPU memory: {gpu_info['memory_total_gb']:.2f} GB total, {gpu_info['memory_free_gb']:.2f} GB free")
        print(f"🔢 CUDA version: {gpu_info['cuda_version']}")
        print(f"🔧 Driver version: {gpu_info['driver_version']}")
        
        # Memory recommendations based on GPU
        if gpu_info['memory_total_gb'] < 16:
            print(f"⚠️ Limited GPU memory detected. Will need to use quantization for larger models.")
            print(f"   Recommended model: 'phi-2' or other smaller models with quantization")
        elif gpu_info['memory_total_gb'] < 40:
            print(f"✅ Sufficient GPU memory for mid-sized models.")
            print(f"   Recommended models: Llama-7B, Mistral-7B with 8-bit quantization")
        else:
            print(f"🚀 Excellent GPU memory. Can run large models without quantization.")
            print(f"   Recommended models: Llama-13B, Mixtral-8x7B, or Llama2-70B with quantization")
    else:
        print(f"⚠️ No GPU detected. Using CPU mode, but processing will be significantly slower.")
        print(f"   Consider using a smaller model like 'phi-2' for CPU inference.")
else:
    # Fallback GPU detection
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"✅ GPU detected: {device_name}")
            print(f"📊 GPU memory: {memory:.2f} GB")
            print(f"🔢 CUDA version: {torch.version.cuda}")
        else:
            print(f"⚠️ No GPU detected. Using CPU mode, but processing will be significantly slower.")
    except ImportError:
        print(f"⚠️ PyTorch not installed, cannot check GPU availability.")
        try:
            gpu_info = subprocess.check_output("nvidia-smi", shell=True).decode('utf-8')
            print(f"✅ GPU detected via nvidia-smi:")
            for line in gpu_info.split('\n')[:10]:  # Show first 10 lines
                if "NVIDIA" in line and not "Driver" in line:
                    print(f"   {line.strip()}")
        except:
            print(f"⚠️ No GPU detected via nvidia-smi. Using CPU mode.")

# %% [markdown]
# ## 3. Dependency Installation and Verification
#
# Let's install and verify the required dependencies for invoice extraction.

# %%
import importlib.util
import subprocess

# Function to check if a package is installed
def is_package_installed(package_name):
    return importlib.util.find_spec(package_name) is not None

# List of required packages
required_packages = [
    "torch", "torchvision", "transformers", "pillow", 
    "pandas", "numpy", "matplotlib", "pyyaml", "tqdm"
]

# Check which packages are installed
print("📦 Checking required packages:")
missing_packages = []
for package in required_packages:
    if is_package_installed(package):
        # Get version if possible
        try:
            module = importlib.import_module(package)
            version = getattr(module, "__version__", "unknown version")
            print(f"✅ {package} ({version})")
        except:
            print(f"✅ {package}")
    else:
        print(f"❌ {package} - not installed")
        missing_packages.append(package)

# Install missing packages if needed
if missing_packages:
    print(f"\n📥 Installing {len(missing_packages)} missing packages...")
    for package in missing_packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {str(e)}")

# Special check for transformers
if is_package_installed("transformers"):
    try:
        import transformers
        print(f"\n🤗 Transformers version: {transformers.__version__}")
        
        # Check if we can load a basic model
        print(f"🧪 Testing transformers model loading capabilities...")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", cache_dir="models/cache/bert-test")
        print(f"✅ Successfully loaded test tokenizer")
    except Exception as e:
        print(f"⚠️ Error loading transformers model: {str(e)}")

# Check PyTorch with CUDA
if is_package_installed("torch"):
    try:
        import torch
        print(f"\n🔥 PyTorch version: {torch.__version__}")
        print(f"🔍 CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"🖥️ CUDA device count: {torch.cuda.device_count()}")
            print(f"🖥️ Current CUDA device: {torch.cuda.current_device()}")
            print(f"🖥️ CUDA device name: {torch.cuda.get_device_name(0)}")
            
            # Test a basic CUDA operation
            print(f"🧪 Testing CUDA tensor operations...")
            a = torch.tensor([1.0, 2.0, 3.0], device="cuda")
            b = torch.tensor([4.0, 5.0, 6.0], device="cuda")
            c = a + b
            print(f"✅ CUDA tensor operation successful: {c.cpu().numpy()}")
    except Exception as e:
        print(f"⚠️ Error testing PyTorch CUDA: {str(e)}")

# %% [markdown]
# ## 4. Path Configuration
#
# Let's set up and validate the paths for data, models, results, and logs.

# %%
import os
from pathlib import Path

# Use our utility module if available
if setup_utils_available:
    paths = configure_paths()
    
    print(f"📂 Path Configuration:")
    for key, path in paths.items():
        status = "✅" if os.path.exists(path) else "❌"
        print(f"{status} {key}: {path}")
        
    # Set environment variables for paths
    for key, path in paths.items():
        env_var = key.upper()
        os.environ[env_var] = str(path)
        print(f"🔄 Set environment variable {env_var}={path}")
else:
    # Fallback path configuration
    # Detect project root
    current_dir = Path.cwd()
    
    # Search for project root by looking for standard directories
    project_root = current_dir
    while project_root != project_root.parent and not all(
        (project_root / d).exists() for d in ["src", "data", "models"]
    ):
        project_root = project_root.parent
    
    if project_root == project_root.parent:
        # If we reached the file system root without finding our markers
        # Use the current directory as project root
        project_root = current_dir
        print("⚠️ Could not detect project root. Using current directory.")
    else:
        print(f"✅ Found project root: {project_root}")
    
    # Define key paths
    paths = {
        "project_root": project_root,
        "data_dir": project_root / "data",
        "models_dir": project_root / "models",
        "results_dir": project_root / "results",
        "logs_dir": project_root / "logs",
        "config_dir": project_root / "configs",
    }
    
    # Check if paths exist
    print(f"\n📂 Path Configuration:")
    for key, path in paths.items():
        os.makedirs(path, exist_ok=True)  # Create if doesn't exist
        status = "✅" if os.path.exists(path) else "❌"
        print(f"{status} {key}: {path}")
        
        # Set environment variables for paths
        env_var = key.upper()
        os.environ[env_var] = str(path)
        print(f"🔄 Set environment variable {env_var}={path}")

# Check if data directory has any invoice images
data_path = Path(os.environ.get("DATA_DIR", "data"))
image_files = list(data_path.glob("**/*.png")) + list(data_path.glob("**/*.jpg")) + list(data_path.glob("**/*.jpeg"))

if image_files:
    print(f"✅ Found {len(image_files)} image files in data directory")
    if len(image_files) > 5:
        print(f"   Sample files: {', '.join([f.name for f in image_files[:5]])}")
    else:
        print(f"   Files: {', '.join([f.name for f in image_files])}")
else:
    print(f"⚠️ No image files found in data directory. Will need to add data before extraction.")

# Check for model files
models_path = Path(os.environ.get("MODELS_DIR", "models"))
model_dirs = [d for d in models_path.glob("*") if d.is_dir()]

if model_dirs:
    print(f"✅ Found {len(model_dirs)} potential model directories")
    for model_dir in model_dirs:
        num_files = len(list(model_dir.glob("*")))
        print(f"   {model_dir.name}: {num_files} files")
else:
    print(f"⚠️ No model directories found. Models will be downloaded on first use.")

# %% [markdown]
# ## 5. Model Availability Check
#
# The invoice extraction system is designed to download and use models on demand. Let's check what models are currently available locally and explain how the model selection system works.

# %%
import os

print("🔍 Model Management System")
print("==========================")
print("The extraction system supports various vision-language models for invoice processing.")
print("Models are downloaded automatically when first used, so you don't need to pre-download them.")
print()

try:
    # Try to use our experiment utils module to list any locally available models
    from src.notebook.experiment_utils import list_available_models
    
    print("📦 Checking for locally available models...")
    models = list_available_models()
    
    if models:
        print(f"✅ Found {len(models)} already downloaded models:")
        for model in models:
            print(f"   📋 {model['name']} ({model['size']:.2f} GB)")
            if 'architecture' in model and model['architecture'] != "Unknown":
                print(f"      Architecture: {model['architecture']}")
    else:
        print("ℹ️ No pre-downloaded models found locally.")
        print("   This is normal - models will be downloaded automatically when you run your first experiment.")
        
    # List recommended models
    print("\n🌟 Recommended Models")
    print("------------------")
    
    # Try to import the model registry if available
    try:
        from src.models.registry import get_available_model_configs
        
        print("These models are configured and ready to use in your experiments:")
        model_configs = get_available_model_configs()
        for model_name, config in model_configs.items():
            memory_req = config.get("memory_requirements", {}).get("gpu_gb", "Unknown")
            model_type = config.get("model_type", "Unknown")
            specialist = "✓" if config.get("is_invoice_specialist", False) else " "
            print(f"   • {model_name}: {model_type} model, ~{memory_req} GB GPU required, Invoice Specialist: {specialist}")
    except ImportError:
        # Fallback recommended models
        print("Common models you can use include:")
        print("   • 'phi-2': Smaller model (2.7B params), works on limited GPU memory or CPU")
        print("   • 'llava-1.5-7b': Mid-sized multimodal model (7B params), good balance of performance and speed")
        print("   • 'llava-1.5-13b': Larger multimodal model (13B params), better performance but requires more GPU memory")
        print("   • 'bakllava-1': Mixture of experts model based on Mixtral, excellent performance but high memory requirements")
        
    print("\n⚙️ How Model Selection Works")
    print("-------------------------")
    print("1. When you create an experiment, you specify which model to use")
    print("2. If the model isn't already downloaded, it will be automatically downloaded")
    print("3. The model is loaded with appropriate optimizations based on your hardware")
    print("4. You can use quantization to reduce memory requirements for larger models")
    
    # Memory requirements guidance
    print("\n💾 Memory Requirements Guide")
    print("-------------------------")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"Your GPU has {gpu_mem:.1f} GB of memory. Based on this:")
            
            if gpu_mem < 8:
                print("   • Recommended: Use smaller models like 'phi-2' with 4-bit quantization")
                print("   • Not recommended: Models larger than 7B parameters")
            elif gpu_mem < 16:
                print("   • Recommended: 7B parameter models with 8-bit quantization")
                print("   • Possible with care: 13B models with 4-bit quantization")
                print("   • Not recommended: Models larger than 13B parameters")
            elif gpu_mem < 32:
                print("   • Recommended: 13B parameter models with 8-bit quantization")
                print("   • Possible with care: Mixture-of-experts models with 4-bit quantization")
            else:
                print("   • Your GPU can handle most models, including large mixture-of-experts models")
                print("   • For best performance, still consider 8-bit quantization for the largest models")
        else:
            print("No GPU detected. When using CPU:")
            print("   • Recommended: Use smaller models like 'phi-2'")
            print("   • Processing will be significantly slower than with a GPU")
    except ImportError:
        print("PyTorch not installed, cannot provide specific memory guidance.")
except ImportError:
    # Fallback basic model discovery
    models_dir = os.environ.get("MODELS_DIR", "models")
    if os.path.exists(models_dir):
        model_dirs = [d for d in os.listdir(models_dir) 
                      if os.path.isdir(os.path.join(models_dir, d))]
        if model_dirs:
            print(f"✅ Found {len(model_dirs)} potential model directories:")
            for model_dir in model_dirs:
                print(f"   📦 {model_dir}")
        else:
            print("ℹ️ No model directories found - models will be downloaded when needed.")
            
        print("\nCommon models you can use include:")
        print("   • 'phi-2': Smaller model, works on limited GPU memory")
        print("   • 'llava-1.5-7b': Mid-sized multimodal model, good balance")
        print("   • 'llava-1.5-13b': Larger model, better performance")
    else:
        print("ℹ️ Models directory not found.")
        print("A models directory will be created when you run your first experiment.")

# %% [markdown]
# ## 6. Data Validation
#
# Let's validate that we have data available for extraction.

# %%
import os
from pathlib import Path
import math

# Find invoice images and ground truth files
data_dir = os.environ.get("DATA_DIR", "data")
image_extensions = ['.png', '.jpg', '.jpeg']

# Search for invoice images
invoice_images = []
for ext in image_extensions:
    invoice_images.extend(Path(data_dir).glob(f"**/*{ext}"))

# Check if we found images
if invoice_images:
    print(f"✅ Found {len(invoice_images)} invoice images")
    
    # Show sample of images if many are found
    if len(invoice_images) > 5:
        sample_size = min(5, len(invoice_images))
        print(f"\n📊 Sample of {sample_size} images:")
        for i, img_path in enumerate(invoice_images[:sample_size]):
            relative_path = os.path.relpath(img_path, data_dir)
            img_size_kb = os.path.getsize(img_path) / 1024
            print(f"   {i+1}. {relative_path} - {img_size_kb:.1f} KB")
    else:
        print("\n📊 Available images:")
        for i, img_path in enumerate(invoice_images):
            relative_path = os.path.relpath(img_path, data_dir)
            img_size_kb = os.path.getsize(img_path) / 1024
            print(f"   {i+1}. {relative_path} - {img_size_kb:.1f} KB")
    
    # Check for ground truth data
    gt_files = list(Path(data_dir).glob("**/*.json")) + list(Path(data_dir).glob("**/*.csv"))
    if gt_files:
        print(f"\n✅ Found {len(gt_files)} potential ground truth files:")
        for i, gt_path in enumerate(gt_files[:3]):  # Show first 3
            relative_path = os.path.relpath(gt_path, data_dir)
            print(f"   {i+1}. {relative_path}")
        if len(gt_files) > 3:
            print(f"   ... and {len(gt_files) - 3} more")
    else:
        print("\n⚠️ No ground truth files found. Evaluation will not be possible.")
else:
    print(f"❌ No invoice images found in {data_dir} directory.")
    print("   Please add some invoice images before running extraction experiments.")

# %% [markdown]
# ## 7. System Readiness Test
#
# Let's run a simple end-to-end test to verify system readiness.

# %%
import time

print("🧪 Running system readiness test...")

tests = [
    ("Directory Structure", lambda: all(os.path.exists(p) for p in [
        os.environ.get("PROJECT_ROOT", ""), 
        os.environ.get("DATA_DIR", "data"),
        os.environ.get("MODELS_DIR", "models"),
        os.environ.get("RESULTS_DIR", "results"),
        os.environ.get("LOGS_DIR", "logs")
    ])),
    
    ("Python Environment", lambda: sys.version_info >= (3, 8)),
    
    ("Core Dependencies", lambda: all(importlib.util.find_spec(pkg) is not None 
                                      for pkg in ["torch", "transformers", "pandas"])),
]

# Add GPU test if applicable
try:
    import torch
    if torch.cuda.is_available():
        tests.append(("GPU Functionality", lambda: torch.tensor([1.0], device="cuda").item() == 1.0))
except ImportError:
    pass

# Run and report test results
all_passed = True
for test_name, test_func in tests:
    try:
        start_time = time.time()
        result = test_func()
        duration = time.time() - start_time
        
        if result:
            print(f"✅ {test_name} - Passed ({duration:.2f}s)")
        else:
            print(f"❌ {test_name} - Failed ({duration:.2f}s)")
            all_passed = False
    except Exception as e:
        print(f"❌ {test_name} - Error: {str(e)}")
        all_passed = False

if all_passed:
    print("\n🎉 All system readiness tests passed! Your environment is ready for invoice extraction.")
else:
    print("\n⚠️ Some system readiness tests failed. Please address the issues before proceeding.")

# %% [markdown]
# ## 8. Troubleshooting Guide
#
# If you encountered issues with the setup, here are some common problems and solutions:

# %% [markdown]
# ### Common Issues
#
# #### GPU Not Detected
# - **Issue**: System doesn't detect your GPU or shows "CUDA not available"
# - **Solutions**:
#   - Ensure you have a compatible NVIDIA GPU
#   - Verify NVIDIA drivers are properly installed (`nvidia-smi` should work)
#   - Check that CUDA is installed and matches your PyTorch version
#   - Try reinstalling PyTorch with the correct CUDA version: `pip install torch==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118`
#
# #### Model Loading Errors
# - **Issue**: Errors when loading models like "file not found" or out of memory
# - **Solutions**:
#   - Check your internet connection for downloading models
#   - Ensure you have enough disk space (models can be several GB)
#   - For CUDA out of memory errors, try a smaller model or enable quantization
#   - Verify model compatibility with your environment
#
# #### Path Configuration Problems
# - **Issue**: Files not found or incorrect paths
# - **Solutions**:
#   - Ensure you're running from the project root directory
#   - Check that all required directories exist
#   - Verify environment variables are correctly set
#   - If using Windows, check for path format issues
#
# #### RunPod-Specific Issues
# - **Issue**: Problems specific to the RunPod environment
# - **Solutions**:
#   - Ensure your pod has enough GPU memory for your selected model
#   - Check disk space in the mounted volume
#   - Verify that required dependencies are installed in the container

# %% [markdown]
# ## 9. Configuration Summary and Export
#
# Let's create a summary of your validated configuration.

# %%
import json
from datetime import datetime

# Collect configuration summary
config_summary = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "environment": {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "in_notebook": in_notebook
    },
    "paths": {
        key: str(path) for key, path in paths.items()
    }
}

# Add GPU information if available
try:
    import torch
    if torch.cuda.is_available():
        config_summary["gpu"] = {
            "device_name": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda,
            "memory_total_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
            "device_count": torch.cuda.device_count()
        }
    else:
        config_summary["gpu"] = {"available": False}
except ImportError:
    config_summary["gpu"] = {"available": "unknown (torch not installed)"}

# Add detected data summary
config_summary["data"] = {
    "image_count": len(invoice_images) if 'invoice_images' in locals() else 0,
}

# Display summary
print("📋 Configuration Summary:")
print(f"🕒 Generated: {config_summary['timestamp']}")
print(f"💻 Platform: {config_summary['environment']['platform']}")
print(f"🐍 Python: {config_summary['environment']['python_version']}")
print(f"📂 Project root: {config_summary['paths']['project_root']}")

if config_summary['gpu'].get('available', False) not in [False, "unknown (torch not installed)"]:
    print(f"🖥️ GPU: {config_summary['gpu']['device_name']}")
    print(f"   CUDA: {config_summary['gpu']['cuda_version']}")
    print(f"   Memory: {config_summary['gpu']['memory_total_gb']:.2f} GB")
else:
    print("🖥️ GPU: Not available")

print(f"🖼️ Invoice images: {config_summary['data']['image_count']}")

# Save configuration summary
os.makedirs(os.environ.get("LOGS_DIR", "logs"), exist_ok=True)
summary_filename = os.path.join(
    os.environ.get("LOGS_DIR", "logs"), 
    f"environment_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
)

with open(summary_filename, 'w') as f:
    json.dump(config_summary, f, indent=2)

print(f"\n✅ Configuration summary saved to: {summary_filename}")

# %% [markdown]
# ## 10. Next Steps
#
# Congratulations! Your environment is now set up and ready for invoice extraction experiments.
#
# Next steps:
#
# 1. Move on to the **Experiment Configuration Notebook** (02_experiment_configuration.ipynb) to:
#    - Configure extraction experiments
#    - Select models and prompts
#    - Run your first extraction pipeline
#
# 2. Check out the **Results Analysis Notebook** (03_results_analysis.ipynb) to:
#    - Analyze extraction results
#    - Compare different models and prompts
#    - Visualize extraction performance
#
# 3. For more in-depth information, explore the documentation in the `docs/` directory:
#    - Setup guide for additional configuration options
#    - Architecture documentation for system understanding
#    - Interface documentation for advanced usage

# %%
print("✅ Environment setup complete!")
print("🚀 You're ready to proceed to the experiment configuration notebook.")

# Optional: For truly confirming readiness, you could run a minimal extraction test here 