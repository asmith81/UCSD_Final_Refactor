"""
Notebook environment setup and validation utilities.

This module provides functions for validating the runtime environment,
checking GPU availability, configuring paths, and ensuring all required
dependencies are installed.
"""

import os
import sys
import subprocess
import platform
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import importlib
import pkg_resources
import json
import shutil
import warnings
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("setup_utils")

# Suppress specific warnings
warnings.filterwarnings("ignore", message="torch.distributed.reduce_op is deprecated")
warnings.filterwarnings("ignore", message="torch._C._jit_set_profiling_executor")

# Import configuration system components
try:
    # Updated imports to reflect new architecture
    from src.config.path_config import get_path_config, get_paths
    from src.config.path_utils import detect_project_root, get_workspace_root
    
    # Try to import environment_config but handle failure gracefully
    try:
        from src.config.environment_config import get_environment_config
        ENV_CONFIG_AVAILABLE = True
    except ImportError as e:
        ENV_CONFIG_AVAILABLE = False
        logger.warning(f"Environment config not available: {e}")
    
    CONFIG_SYSTEM_AVAILABLE = True
    logger.info("Configuration system available, using integrated approach")
except ImportError as e:
    CONFIG_SYSTEM_AVAILABLE = False
    ENV_CONFIG_AVAILABLE = False
    logger.warning(f"Configuration system not available: {e}")
    logger.warning("Using standalone implementation for notebook setup")


def validate_environment() -> Dict[str, bool]:
    """
    Validate the runtime environment for compatibility with the extraction system.
    
    Returns:
        Dictionary with validation results for different components
    """
    results = {
        "python_version": False,
        "cuda_available": False,
        "required_packages": False,
        "directory_structure": False,
        "environment_variables": False,
        "disk_space": False
    }
    
    # Check Python version
    py_version = sys.version_info
    results["python_version"] = (py_version.major == 3 and py_version.minor >= 8)
    logger.info(f"Python version: {py_version.major}.{py_version.minor}.{py_version.micro} - {'✓' if results['python_version'] else '✗'}")
    
    # Check CUDA availability
    try:
        import torch
        results["cuda_available"] = torch.cuda.is_available()
        
        if results["cuda_available"]:
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            logger.info(f"CUDA available: {device_name}, CUDA version: {cuda_version} - ✓")
        else:
            logger.warning("CUDA not available. GPU acceleration will not be used.")
    except ImportError:
        logger.warning("PyTorch not installed. Cannot check CUDA availability.")
    
    # Check required packages
    required_packages = [
        "torch", "numpy", "pandas", "matplotlib", 
        "tqdm", "pyyaml", "pillow", "transformers"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            missing_packages.append(package)
    
    results["required_packages"] = len(missing_packages) == 0
    if results["required_packages"]:
        logger.info("All required packages installed - ✓")
    else:
        logger.warning(f"Missing required packages: {', '.join(missing_packages)}")
    
    # Check directory structure
    base_dir = Path(os.getcwd())
    if not (base_dir / "src").exists():
        # Try to find project root
        if (base_dir.parent / "src").exists():
            base_dir = base_dir.parent
    
    required_dirs = ["src", "data", "models", "configs", "notebooks", "results"]
    missing_dirs = [d for d in required_dirs if not (base_dir / d).exists()]
    
    results["directory_structure"] = len(missing_dirs) == 0
    if results["directory_structure"]:
        logger.info("Directory structure valid - ✓")
    else:
        logger.warning(f"Missing directories: {', '.join(missing_dirs)}")
    
    # Check environment variables
    env_vars = ["DATA_DIR", "MODELS_DIR", "RESULTS_DIR", "CONFIGS_DIR"]
    missing_vars = [var for var in env_vars if not os.environ.get(var)]
    
    if not missing_vars:
        results["environment_variables"] = True
        logger.info("Environment variables configured - ✓")
    else:
        logger.warning(f"Missing environment variables: {', '.join(missing_vars)}")
        logger.info("Will try to set these from default paths.")
    
    # Check disk space
    try:
        free_space = shutil.disk_usage("/").free / (1024 ** 3)  # Free space in GB
        results["disk_space"] = free_space >= 10  # Require at least 10GB free
        logger.info(f"Free disk space: {free_space:.1f} GB - {'✓' if results['disk_space'] else '✗'}")
    except:
        logger.warning("Cannot check disk space")
    
    return results


def check_gpu_availability() -> Dict[str, Any]:
    """
    Check GPU availability and provide detailed information.
    
    This function uses the environment configuration when available,
    with a fallback to the standalone implementation.
    
    Returns:
        Dictionary with GPU information
    """
    if CONFIG_SYSTEM_AVAILABLE and ENV_CONFIG_AVAILABLE:
        try:
            # Try to use the environment configuration
            logger.info("Using environment configuration for GPU check")
            env_config = get_environment_config()
            
            # Get GPU info from environment config
            gpu_info = {
                "available": env_config._is_cuda_available(),
                "name": "Unknown",
                "memory_total": 0,
                "memory_total_gb": 0,
                "memory_free": 0,
                "memory_free_gb": 0,
                "cuda_version": "Unknown",
                "device_count": 0,
                "driver_version": "Unknown"
            }
            
            # Try to get more detailed info if GPU is available
            if gpu_info["available"]:
                try:
                    import torch
                    if torch.cuda.is_available():
                        gpu_info["name"] = torch.cuda.get_device_name(0)
                        gpu_info["device_count"] = torch.cuda.device_count()
                        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                        gpu_info["memory_total_gb"] = total_mem
                        gpu_info["memory_total"] = total_mem
                        # Estimate free memory
                        gpu_info["memory_free_gb"] = total_mem  # Simplification
                        gpu_info["memory_free"] = total_mem     # Simplification
                        gpu_info["cuda_version"] = torch.version.cuda
                except Exception as e:
                    logger.warning(f"Could not get detailed GPU info: {e}")
            
            if gpu_info["available"]:
                logger.info(f"GPU: {gpu_info['name']}")
                logger.info(f"CUDA Version: {gpu_info['cuda_version']}")
                logger.info(f"Total Memory: {gpu_info['memory_total_gb']:.2f} GB")
                logger.info(f"Free Memory: {gpu_info['memory_free_gb']:.2f} GB")
            
            return gpu_info
            
        except Exception as e:
            logger.warning(f"Error using environment configuration for GPU check: {e}")
            logger.warning("Falling back to standalone implementation")
    
    # Standalone implementation (original code)
    logger.info("Using standalone implementation for GPU check")
    
    gpu_info = {
        "available": False,
        "name": "Not available",
        "memory_total": 0,
        "memory_free": 0,
        "memory_total_gb": 0.0,
        "memory_free_gb": 0.0,
        "cuda_version": None,
        "device_count": 0,
        "driver_version": "None"
    }
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_info["available"] = True
            gpu_info["device_count"] = torch.cuda.device_count()
            gpu_info["name"] = torch.cuda.get_device_name(0)
            gpu_info["cuda_version"] = torch.version.cuda
            
            # Get memory information
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
            gpu_info["memory_total"] = total_memory
            gpu_info["memory_total_gb"] = total_memory
            
            # Reserved memory calculation (rough estimate)
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            reserved = torch.cuda.memory_reserved(0) / (1024**3)
            free_memory = total_memory - reserved
            gpu_info["memory_free"] = free_memory
            gpu_info["memory_free_gb"] = free_memory
            
            # Try to get driver version
            try:
                gpu_info["driver_version"] = torch.version.cuda
                # On some systems we might be able to get the actual driver version
                if hasattr(torch.cuda, 'get_driver_version'):
                    gpu_info["driver_version"] = torch.cuda.get_driver_version()
            except:
                pass
            
            logger.info(f"GPU: {gpu_info['name']}")
            logger.info(f"CUDA Version: {gpu_info['cuda_version']}")
            logger.info(f"Total Memory: {gpu_info['memory_total_gb']:.2f} GB")
            logger.info(f"Free Memory: {gpu_info['memory_free_gb']:.2f} GB")
        else:
            logger.warning("No GPU available")
    except ImportError:
        logger.warning("PyTorch not installed. Cannot check GPU availability.")
    except Exception as e:
        logger.warning(f"Error checking GPU: {str(e)}")
    
    return gpu_info


def configure_paths(
    data_dir: Optional[str] = None,
    models_dir: Optional[str] = None,
    results_dir: Optional[str] = None,
    configs_dir: Optional[str] = None
) -> Dict[str, str]:
    """
    Configure paths for the invoice extraction system.
    
    This function sets up paths using the path configuration system when available,
    with a fallback to a standalone implementation.
    
    Args:
        data_dir: Optional path to data directory
        models_dir: Optional path to models directory
        results_dir: Optional path to results directory
        configs_dir: Optional path to configuration directory
        
    Returns:
        Dictionary with configured paths
    """
    # Create custom paths dictionary if any paths were provided
    custom_paths = {}
    if data_dir: custom_paths['data_dir'] = data_dir
    if models_dir: custom_paths['models_dir'] = models_dir
    if results_dir: custom_paths['results_dir'] = results_dir
    if configs_dir: custom_paths['config_dir'] = configs_dir
    
    if CONFIG_SYSTEM_AVAILABLE:
        try:
            # Try to use the path configuration system
            logger.info("Using path configuration system")
            
            # Initialize or reset the path configuration with custom paths
            path_config = get_path_config(
                custom_paths=custom_paths if custom_paths else None,
                reset=True
            )
            
            # Get all paths
            paths = get_paths()
            
            # Set environment variables for configured paths
            for key, path in paths.items():
                env_var = key.upper()
                os.environ[env_var] = str(path)
                logger.debug(f"Set environment variable {env_var}={path}")
            
            # Log configured paths
            logger.info(f"Configured paths:")
            for key in ['project_root', 'data_dir', 'models_dir', 'results_dir', 'config_dir']:
                if key in paths:
                    exists = os.path.exists(paths[key])
                    status = "✓" if exists else "✗"
                    logger.info(f"{status} {key}: {paths[key]}")
            
            return paths
            
        except Exception as e:
            logger.warning(f"Error using path configuration system: {e}")
            logger.warning("Falling back to standalone implementation")
    
    # Standalone implementation (original code)
    logger.info("Using standalone implementation for path configuration")
    
    # Determine base directory (project root)
    base_dir = Path(os.getcwd())
    if not (base_dir / "src").exists():
        if (base_dir.parent / "src").exists():
            base_dir = base_dir.parent
    
    # In RunPod, use /workspace as project root if available
    if os.path.exists("/workspace"):
        base_dir = Path("/workspace")
    
    # Set default paths if not provided
    if not data_dir:
        data_dir = str(base_dir / "data")
    if not models_dir:
        models_dir = str(base_dir / "models")
    if not results_dir:
        results_dir = str(base_dir / "results")
    if not configs_dir:
        configs_dir = str(base_dir / "configs")
    
    # Create directories if they don't exist
    Path(data_dir).mkdir(exist_ok=True)
    Path(models_dir).mkdir(exist_ok=True)
    Path(results_dir).mkdir(exist_ok=True)
    Path(configs_dir).mkdir(exist_ok=True)
    
    # Set environment variables
    os.environ["PROJECT_ROOT"] = str(base_dir)
    os.environ["DATA_DIR"] = data_dir
    os.environ["MODELS_DIR"] = models_dir
    os.environ["RESULTS_DIR"] = results_dir
    os.environ["CONFIGS_DIR"] = configs_dir
    os.environ["LOGS_DIR"] = str(base_dir / "logs")
    
    # Set environment variable to indicate we're in a notebook environment
    os.environ["IN_NOTEBOOK"] = "1"
    
    paths = {
        "project_root": str(base_dir),
        "base_dir": str(base_dir),
        "data_dir": data_dir,
        "models_dir": models_dir,
        "results_dir": results_dir,
        "configs_dir": configs_dir,
        "logs_dir": str(base_dir / "logs")
    }
    
    logger.info("Paths configured:")
    for key, path in paths.items():
        logger.info(f"  {key}: {path}")
    
    # Create .env file for future runs
    env_file = base_dir / ".env"
    with open(env_file, "w") as f:
        f.write(f"PROJECT_ROOT={str(base_dir)}\n")
        for key, path in paths.items():
            if key not in ["base_dir", "project_root"]:
                f.write(f"{key.upper()}={path}\n")
        f.write("USE_GPU=True\n")
    
    logger.info(f"Environment configuration saved to {env_file}")
    
    return paths


def install_dependencies(
    requirements: Optional[List[str]] = None,
    upgrade: bool = False
) -> bool:
    """
    Install required dependencies.
    
    Args:
        requirements: Optional list of requirements to install
        upgrade: Whether to upgrade existing packages
        
    Returns:
        True if installation was successful
    """
    if requirements is None:
        # Default requirements
        requirements = [
            "torch>=2.0.0",
            "numpy>=1.20.0",
            "pandas>=1.3.0",
            "matplotlib>=3.4.0",
            "tqdm>=4.60.0",
            "pyyaml>=6.0",
            "pillow>=8.2.0",
            "transformers>=4.20.0",
            "ipywidgets>=7.7.0",
            "jupyterlab>=3.0.0"
        ]
    
    logger.info("Installing dependencies...")
    
    # First check which packages need installation
    to_install = []
    for req in requirements:
        package = req.split(">=")[0].split("==")[0].split("<")[0].strip()
        try:
            pkg_resources.get_distribution(package)
            if upgrade:
                to_install.append(req)
        except pkg_resources.DistributionNotFound:
            to_install.append(req)
    
    if not to_install:
        logger.info("All dependencies already installed.")
        return True
    
    # Install required packages
    cmd = [sys.executable, "-m", "pip", "install"]
    if upgrade:
        cmd.append("--upgrade")
    cmd.extend(to_install)
    
    try:
        logger.info(f"Running: {' '.join(cmd)}")
        subprocess.check_call(cmd)
        logger.info("Dependencies installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing dependencies: {str(e)}")
        return False


def test_model_loading(model_name: str = "small", model_repo: Optional[str] = None, test_inputs: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Test loading a model to verify setup.
    
    Args:
        model_name: Name of the model to test load (small/medium/large)
        model_repo: Optional repository ID to use for loading the model
        test_inputs: Optional list of test inputs for the model
        
    Returns:
        Dictionary with test results
    """
    results = {
        "success": False,
        "model_loaded": False,
        "memory_used": 0,
        "load_time": 0,
        "error": None
    }
    
    logger.info(f"Testing model loading: {model_name}")
    
    try:
        import torch
        import time
        from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
        
        start_time = time.time()
        
        # Based on model_name, choose a small/medium model for testing
        if model_name == "small":
            model_id = "google/flan-t5-small"  # ~300M parameters
        elif model_name == "medium":
            model_id = "google/flan-t5-base"   # ~800M parameters
        elif model_name == "large":
            model_id = "google/flan-t5-large"  # ~3B parameters
        else:
            model_id = model_name  # Use provided model name
        
        # Load model and tokenizer
        logger.info(f"Loading model: {model_id}")
        
        # Check GPU availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Track memory before loading
        if device == "cuda":
            torch.cuda.empty_cache()
            mem_before = torch.cuda.memory_allocated() / (1024**3)
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
        model.to(device)
        
        # Calculate stats
        load_time = time.time() - start_time
        
        if device == "cuda":
            mem_after = torch.cuda.memory_allocated() / (1024**3)
            memory_used = mem_after - mem_before
        else:
            memory_used = 0
        
        # Run a simple prediction to ensure model works
        inputs = tokenizer("Hello, I am testing the extraction system.", return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=20)
        response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Update results
        results["success"] = True
        results["model_loaded"] = True
        results["memory_used"] = memory_used
        results["load_time"] = load_time
        results["device"] = device
        
        logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
        logger.info(f"Memory used: {memory_used:.2f} GB")
        logger.info(f"Device: {device}")
        
        # Clean up
        del model, tokenizer
        if device == "cuda":
            torch.cuda.empty_cache()
            
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        results["error"] = str(e)
    
    return results


def setup_notebook_environment(
    configure_system: bool = True,
    verify_dependencies: bool = True,
    verify_gpu: bool = True,
    verify_model_loading: bool = False,
    model_name: str = "pixtral-12b",
    model_repo: Optional[str] = None,
    test_inputs: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Set up the notebook environment for extraction tasks.
    
    This function integrates with the configuration system when available,
    with fallbacks to standalone implementations.
    
    Args:
        configure_system: Whether to configure paths and environment variables
        verify_dependencies: Whether to verify and potentially install dependencies
        verify_gpu: Whether to verify GPU availability
        verify_model_loading: Whether to test loading the model
        model_name: Name of the model to test loading
        model_repo: Optional repository ID to use for loading the model
        test_inputs: Optional list of test inputs for the model
        
    Returns:
        Dictionary with setup results
    """
    logger.info("Setting up notebook environment")
    
    # Track setup progress
    setup_results = {
        "environment_validated": False,
        "paths_configured": False,
        "gpu_available": False,
        "dependencies_installed": False,
        "model_loaded": False,
        "system_info": None
    }
    
    if CONFIG_SYSTEM_AVAILABLE:
        try:
            # Try to use the configuration system for setup
            logger.info("Using configuration system for environment setup")
            
            # Get configuration manager and environment configuration
            config_manager = get_config_manager()
            env_config = get_environment_config()
            
            # Configure paths if requested
            if configure_system:
                # Configure using path_config
                path_config = get_path_config()
                setup_results["paths_configured"] = True
                
                # Extract paths for backward compatibility
                paths = {
                    "project_root": str(path_config.get_path('project_root') or '/workspace'),
                    "data_dir": str(path_config.get_path('data_dir')),
                    "models_dir": str(path_config.get_path('models_dir')),
                    "results_dir": str(path_config.get_path('results_dir')),
                    "configs_dir": str(path_config.get_path('configs_dir')),
                    "logs_dir": str(path_config.get_path('logs_dir')),
                    "working_dir": os.getcwd()
                }
                setup_results["paths"] = paths
            
            # Verify GPU if requested
            if verify_gpu:
                # Check GPU using environment config
                gpu_info = check_gpu_availability()
                setup_results["gpu_available"] = gpu_info["available"]
                setup_results["gpu_info"] = gpu_info
            
            # Install dependencies if requested
            if verify_dependencies:
                # Check dependencies using environment config
                dependencies = env_config.dependencies
                if dependencies.missing_packages:
                    logger.warning(f"Missing dependencies: {dependencies.missing_packages}")
                    # Install missing dependencies
                    setup_results["dependencies_installed"] = install_dependencies(dependencies.missing_packages)
                else:
                    setup_results["dependencies_installed"] = True
            
            # Verify model loading if requested
            if verify_model_loading:
                try:
                    # Use model configuration from config manager
                    model_config = config_manager.get_model_config(model_name)
                    if model_config:
                        model_repo = model_repo or model_config.repo_id
                    
                    # Test model loading
                    model_results = test_model_loading(
                        model_name=model_name,
                        model_repo=model_repo,
                        test_inputs=test_inputs
                    )
                    setup_results["model_loaded"] = model_results["success"]
                    setup_results["model_results"] = model_results
                except Exception as e:
                    logger.error(f"Error loading model: {e}")
                    setup_results["model_loaded"] = False
                    setup_results["model_error"] = str(e)
            
            # Get system information
            setup_results["system_info"] = get_system_info()
            
            # Mark environment as validated
            setup_results["environment_validated"] = True
            return setup_results
            
        except Exception as e:
            logger.warning(f"Error using configuration system for setup: {e}")
            logger.warning("Falling back to standalone implementation")
    
    # Standalone implementation (original code)
    logger.info("Using standalone implementation for environment setup")
    
    # Validate basic environment
    validate_results = validate_environment()
    setup_results["environment_validated"] = all(validate_results.values())
    
    # Configure paths if requested
    if configure_system:
        paths = configure_paths()
        setup_results["paths_configured"] = True
        setup_results["paths"] = paths
    
    # Check GPU availability if requested
    if verify_gpu:
        gpu_info = check_gpu_availability()
        setup_results["gpu_available"] = gpu_info["available"]
        setup_results["gpu_info"] = gpu_info
    
    # Install dependencies if requested
    if verify_dependencies:
        dep_result = install_dependencies()
        setup_results["dependencies_installed"] = dep_result
    
    # Test model loading if requested
    if verify_model_loading:
        try:
            model_results = test_model_loading(
                model_name=model_name,
                model_repo=model_repo,
                test_inputs=test_inputs
            )
            setup_results["model_loaded"] = model_results["success"]
            setup_results["model_results"] = model_results
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            setup_results["model_loaded"] = False
            setup_results["model_error"] = str(e)
    
    # Get system information
    setup_results["system_info"] = get_system_info()
    
    return setup_results


def get_system_info() -> Dict[str, Any]:
    """
    Get detailed information about the system.
    
    Returns:
        Dictionary with system information
    """
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_path": sys.executable,
        "processor": platform.processor(),
        "memory": None,
        "gpu": None,
        "packages": {},
        "env_vars": {},
        "disk_space": {"total_gb": 0, "free_gb": 0, "used_gb": 0},
        "in_runpod": False
    }
    
    # Get RAM information
    try:
        import psutil
        vm = psutil.virtual_memory()
        info["memory"] = {
            "total": vm.total / (1024**3),
            "available": vm.available / (1024**3),
            "percent_used": vm.percent
        }
    except ImportError:
        pass
    
    # Get GPU information
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "count": torch.cuda.device_count(),
                "cuda_version": torch.version.cuda,
                "memory_total": torch.cuda.get_device_properties(0).total_memory / (1024**3)
            }
    except ImportError:
        pass
    
    # Get package versions
    important_packages = [
        "torch", "numpy", "pandas", "matplotlib", 
        "transformers", "pillow", "tqdm", "pyyaml"
    ]
    
    for package in important_packages:
        try:
            version = pkg_resources.get_distribution(package).version
            info["packages"][package] = version
        except:
            info["packages"][package] = "not installed"
    
    # Get environment variables that might be useful
    for env_var in ["CUDA_VISIBLE_DEVICES", "RUNPOD_POD_ID", "GPU_NAME"]:
        if env_var in os.environ:
            info["env_vars"][env_var] = os.environ[env_var]
    
    # Check for RunPod environment
    info["in_runpod"] = "RUNPOD_POD_ID" in os.environ or "H100" in os.environ.get("GPU_NAME", "")
    
    # Get disk space information
    try:
        total, used, free = shutil.disk_usage("/")
        info["disk_space"] = {
            "total_gb": total / (1024**3),
            "used_gb": used / (1024**3),
            "free_gb": free / (1024**3)
        }
    except:
        pass
    
    return info


def get_config_summary() -> Dict[str, Any]:
    """
    Generate a configuration summary for notebooks.
    
    This function uses the path configuration to get a structured configuration
    summary when available, with a fallback to the standalone implementation.
    
    Returns:
        Dictionary with organized configuration summary
    """
    if CONFIG_SYSTEM_AVAILABLE:
        try:
            # Try to use the path_config system
            logger.info("Using Path Configuration for configuration summary")
            
            # Get paths from the path configuration
            paths = get_paths()
            
            # Get system information
            sys_info = get_system_info()
            
            # Build the summary
            summary = {
                "environment": {
                    "platform": sys_info["platform"],
                    "python_version": sys_info["python_version"],
                    "python_implementation": sys_info["python_implementation"],
                    "in_notebook": "IN_NOTEBOOK" in os.environ,
                    "in_runpod": sys_info["in_runpod"]
                },
                "paths": paths,
                "gpu": sys_info["gpu"],
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Try to add environment config if available
            if ENV_CONFIG_AVAILABLE:
                try:
                    env_config = get_environment_config()
                    summary["environment"].update({
                        "env_type": env_config.env_type,
                        "env_name": env_config.name
                    })
                except Exception as e:
                    logger.warning(f"Could not add environment config details: {e}")
            
            logger.info("Generated configuration summary")
            return summary
        
        except Exception as e:
            logger.warning(f"Error generating configuration summary: {e}")
            logger.warning("Falling back to standalone implementation")
    
    # Standalone implementation (original code)
    logger.info("Using standalone implementation for configuration summary")
    
    # Fall back to the original implementation
    sys_info = get_system_info()
    
    # Determine project root
    project_root = os.environ.get("PROJECT_ROOT", "/workspace")
    if not os.path.exists(project_root):
        if os.path.exists("/workspace"):
            project_root = "/workspace"
        else:
            project_root = os.getcwd()
    
    # Structure for config summary
    summary = {
        "environment": {
            "platform": sys_info["platform"],
            "python_version": sys_info["python_version"],
            "python_implementation": sys_info["python_implementation"],
            "in_notebook": "IN_NOTEBOOK" in os.environ,
            "in_runpod": sys_info["in_runpod"]
        },
        "paths": {
            "project_root": project_root,
            "data_dir": os.environ.get("DATA_DIR", os.path.join(project_root, "data")),
            "models_dir": os.environ.get("MODELS_DIR", os.path.join(project_root, "models")),
            "results_dir": os.environ.get("RESULTS_DIR", os.path.join(project_root, "results")),
            "configs_dir": os.environ.get("CONFIGS_DIR", os.path.join(project_root, "configs")),
            "working_dir": os.getcwd()
        },
        "gpu": {
            "available": False,
            "device_name": "None",
            "memory_gb": 0.0,
            "cuda_version": "None"
        },
        "packages": sys_info["packages"],
        "disk_space": sys_info["disk_space"]
    }
    
    # Fill in GPU info if available
    if sys_info["gpu"] is not None:
        summary["gpu"] = {
            "available": True,
            "device_name": sys_info["gpu"]["name"],
            "memory_gb": sys_info["gpu"]["memory_total"],
            "cuda_version": sys_info["gpu"]["cuda_version"]
        }
    
    return summary


def export_config_summary(summary: Dict[str, Any], export_path: Optional[str] = None) -> str:
    """
    Export configuration summary to a JSON file.
    
    This function uses the ConfigurationManager to export the configuration when available,
    with a fallback to the standalone implementation.
    
    Args:
        summary: Configuration summary dictionary
        export_path: Path to export the summary (if None, a default path is used)
        
    Returns:
        Path to the exported file
    """
    if CONFIG_SYSTEM_AVAILABLE:
        try:
            # Try to use the ConfigurationManager for export
            logger.info("Using ConfigurationManager for export")
            
            if export_path is not None:
                return get_config_manager().export_notebook_config(export_path)
            else:
                # If summary was provided but not generated by ConfigurationManager,
                # still use it with our own export logic
                if summary is not None:
                    # Fallback to direct export below
                    raise ValueError("Using provided summary with fallback export")
                return get_config_manager().export_notebook_config()
        
        except Exception as e:
            logger.warning(f"Error using ConfigurationManager for export: {e}")
            logger.warning("Falling back to standalone implementation")
    
    # Standalone implementation (original code)
    logger.info("Using standalone implementation for export")
    
    if export_path is None:
        # Get the project root as the base for the export
        project_root = summary["paths"]["project_root"]
        # Use results directory if available, otherwise use project root
        results_dir = os.environ.get("RESULTS_DIR", os.path.join(project_root, "results"))
        os.makedirs(results_dir, exist_ok=True)
        # Create a timestamped filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        export_path = os.path.join(results_dir, f"config_summary_{timestamp}.json")
    
    # Export as JSON
    with open(export_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Configuration summary exported to: {export_path}")
    return export_path


def get_notebook_paths() -> Dict[str, str]:
    """
    Get path dictionary configured for use in notebooks.
    
    Returns:
        Dictionary with configured paths, including project_root
    """
    # Determine project root
    project_root = os.environ.get("PROJECT_ROOT")
    if not project_root or not os.path.exists(project_root):
        # Try using /workspace for RunPod
        if os.path.exists("/workspace"):
            project_root = "/workspace"
        else:
            current_dir = os.getcwd()
            # Look for src directory
            if os.path.exists(os.path.join(current_dir, "src")):
                project_root = current_dir
            elif os.path.exists(os.path.join(os.path.dirname(current_dir), "src")):
                project_root = os.path.dirname(current_dir)
            else:
                # Fallback to current directory
                project_root = current_dir
    
    # Set as environment variable
    os.environ["PROJECT_ROOT"] = project_root
    
    # Configure paths
    paths = {
        "project_root": project_root,
        "data_dir": os.environ.get("DATA_DIR", os.path.join(project_root, "data")),
        "models_dir": os.environ.get("MODELS_DIR", os.path.join(project_root, "models")),
        "results_dir": os.environ.get("RESULTS_DIR", os.path.join(project_root, "results")),
        "configs_dir": os.environ.get("CONFIGS_DIR", os.path.join(project_root, "configs")),
        "logs_dir": os.environ.get("LOGS_DIR", os.path.join(project_root, "logs")),
        "working_dir": os.getcwd()
    }
    
    # Ensure directories exist
    for key, path in paths.items():
        if key != "working_dir" and key != "project_root":
            os.makedirs(path, exist_ok=True)
    
    return paths


def fix_notebook_paths(paths_dict):
    """
    Fix paths dictionary for notebook by ensuring project_root is set.
    
    This function can be called directly in a notebook cell to fix path issues.
    
    Args:
        paths_dict: The existing paths dictionary to fix
        
    Returns:
        Updated paths dictionary with project_root set
    """
    # Make a copy of the dictionary to avoid modifying the original
    fixed_paths = paths_dict.copy() if paths_dict else {}
    
    # Ensure project_root is set
    if 'project_root' not in fixed_paths or not fixed_paths['project_root']:
        # Get project root from environment or use a fallback
        project_root = os.environ.get("PROJECT_ROOT")
        if not project_root or not os.path.exists(project_root):
            # Try using /workspace for RunPod
            if os.path.exists("/workspace"):
                project_root = "/workspace"
            else:
                # Fallback to current directory or parent
                current_dir = os.getcwd()
                if os.path.exists(os.path.join(current_dir, "src")):
                    project_root = current_dir
                elif os.path.exists(os.path.join(os.path.dirname(current_dir), "src")):
                    project_root = os.path.dirname(current_dir)
                else:
                    project_root = current_dir
        
        # Set project_root in the paths dictionary
        fixed_paths['project_root'] = project_root
        
        # Log the fix
        logger.info(f"Fixed missing project_root in paths: {project_root}")
    
    # Ensure all standard path keys exist
    standard_paths = ["data_dir", "models_dir", "results_dir", "configs_dir", "logs_dir"]
    project_root = fixed_paths['project_root']
    
    for path_key in standard_paths:
        if path_key not in fixed_paths or not fixed_paths[path_key]:
            # Create standard path from project_root
            fixed_paths[path_key] = os.path.join(project_root, path_key.replace('_dir', ''))
            logger.info(f"Added missing {path_key}: {fixed_paths[path_key]}")
    
    return fixed_paths


def _is_runpod_env() -> bool:
    """Check if running in RunPod environment."""
    if any(var in os.environ for var in ["RUNPOD_POD_ID", "RUNPOD_API_KEY"]):
        return True
    if Path("/workspace").exists() and Path("/runpod").exists():
        return True
    return False


if __name__ == "__main__":
    # If executed as a script, run the complete setup
    results = setup_notebook_environment()
    print(json.dumps(results, indent=2)) 