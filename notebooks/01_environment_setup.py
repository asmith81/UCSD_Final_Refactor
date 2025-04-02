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
# 3. Verified model availability and compatibility
# 4. Tested the system with a simple extraction task
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
# ## 4. Model Availability Check
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
        print("ℹ️ No models found in registry or config folder.")
        print("   Please add model configuration files to configs/models/ directory or install models to your models directory.")
        
    # List recommended models
    print("\n🌟 Recommended Models")
    print("------------------")
    
    # Try to import the model registry if available
    try:
        from src.models.registry import get_available_model_configs
        
        model_configs = get_available_model_configs()
        if model_configs:
            print("These models are configured and ready to use in your experiments:")
            for model_name, config in model_configs.items():
                memory_req = config.get("memory_requirements", {}).get("gpu_gb", "Unknown")
                model_type = config.get("model_type", "Unknown")
                specialist = "✓" if config.get("is_invoice_specialist", False) else " "
                print(f"   • {model_name}: {model_type} model, ~{memory_req} GB GPU required, Invoice Specialist: {specialist}")
        else:
            print("⚠️ No models configured in the registry.")
            print("   Please add model configuration files to the configs/models/ directory.")
    except ImportError:
        print("⚠️ Model registry not available.")
        print("   To properly configure models, please ensure the src/models/registry.py module is accessible.")
    
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
            print("⚠️ No models found in local directories.")
            print("   Please add model configuration files to configs/models/ directory or install models to your models directory.")
            
        print("\n⚠️ Model registry not available.")
        print("   To properly configure models, please ensure the src/models/registry.py module is accessible.")

# %% [markdown]
# ## 5. System Readiness Test
#
# Let's run a simple end-to-end test to verify system readiness.

# %%
import time

print("🧪 Running system readiness test...")

tests = [
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
# ## 6. Next Steps
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