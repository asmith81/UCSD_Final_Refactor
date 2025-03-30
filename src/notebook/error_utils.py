"""
Error handling utilities for notebooks.

This module provides enhanced error handling, logging, and debugging
capabilities specifically optimized for notebook environments. It includes
categorized error messages, visual formatting, and RunPod-specific 
diagnostic information.
"""

import os
import sys
import logging
import traceback
import platform
import subprocess
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("error_utils")

# Error categories with descriptive information
ERROR_CATEGORIES = {
    "environment": {
        "title": "Environment Error",
        "description": "Issues with system configuration, dependencies, or resources",
        "icon": "🌍"
    },
    "model": {
        "title": "Model Error",
        "description": "Issues with model loading, initialization, or compatibility",
        "icon": "🤖"
    },
    "data": {
        "title": "Data Error",
        "description": "Issues with data loading, processing, or validation",
        "icon": "📊"
    },
    "config": {
        "title": "Configuration Error",
        "description": "Issues with experiment or pipeline configuration",
        "icon": "⚙️"
    },
    "execution": {
        "title": "Execution Error",
        "description": "Issues during pipeline or experiment execution",
        "icon": "⚡"
    },
    "gpu": {
        "title": "GPU Error",
        "description": "Issues with GPU allocation, CUDA compatibility, or memory",
        "icon": "🖥️"
    },
    "runpod": {
        "title": "RunPod Error",
        "description": "Issues specific to the RunPod environment",
        "icon": "☁️"
    },
    "internal": {
        "title": "Internal Error",
        "description": "Unexpected system errors or bugs",
        "icon": "⚠️"
    }
}


class NotebookFriendlyError(Exception):
    """
    Exception class with notebook-friendly formatting and diagnostic information.
    """
    def __init__(
        self, 
        message: str, 
        category: str = "internal",
        original_error: Optional[Exception] = None,
        recovery_steps: Optional[List[str]] = None,
        debug_info: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.category = category.lower()
        self.original_error = original_error
        self.recovery_steps = recovery_steps or []
        self.debug_info = debug_info or {}
        self.timestamp = datetime.now()
        
        # Construct the formatted message
        category_info = ERROR_CATEGORIES.get(self.category, ERROR_CATEGORIES["internal"])
        formatted_message = (
            f"{category_info['icon']} {category_info['title']}: {message}\n"
        )
        
        if original_error:
            formatted_message += f"🔍 Original error: {str(original_error)}\n"
        
        if recovery_steps:
            formatted_message += "🔄 Recovery steps:\n"
            for i, step in enumerate(recovery_steps, 1):
                formatted_message += f"   {i}. {step}\n"
        
        super().__init__(formatted_message)
        
        # Log the error
        logger.error(f"{category_info['title']}: {message}")
        if original_error:
            logger.error(f"Original error: {str(original_error)}")


def is_runpod() -> bool:
    """
    Check if we're running in a RunPod environment.
    
    Returns:
        bool: True if in RunPod environment, False otherwise
    """
    return (
        os.environ.get("RUNPOD_POD_ID") is not None or 
        "A100" in os.environ.get("GPU_NAME", "") or
        "H100" in os.environ.get("GPU_NAME", "")
    )


def get_gpu_info() -> Dict[str, Any]:
    """
    Get detailed information about the GPU environment.
    
    Returns:
        Dict with GPU details or empty dict if no GPU
    """
    result = {
        "available": False,
        "name": "None",
        "memory_total_gb": 0,
        "memory_used_gb": 0, 
        "cuda_version": "None",
        "driver_version": "None"
    }
    
    try:
        import torch
        
        result["available"] = torch.cuda.is_available()
        if result["available"]:
            # Get device name and memory
            device_id = 0
            result["name"] = torch.cuda.get_device_name(device_id)
            result["device_count"] = torch.cuda.device_count()
            
            # Get memory information
            props = torch.cuda.get_device_properties(device_id)
            result["memory_total_gb"] = props.total_memory / (1024**3)
            result["memory_used_gb"] = torch.cuda.memory_allocated(device_id) / (1024**3)
            result["memory_free_gb"] = result["memory_total_gb"] - result["memory_used_gb"]
            
            # Get CUDA version
            result["cuda_version"] = torch.version.cuda
            
            # Try to get driver version via nvidia-smi
            try:
                nvidia_smi = subprocess.check_output("nvidia-smi", shell=True).decode('utf-8')
                for line in nvidia_smi.split('\n'):
                    if "Driver Version:" in line:
                        result["driver_version"] = line.split("Driver Version:")[1].strip().split()[0]
                        break
            except:
                pass
    except ImportError:
        logger.warning("PyTorch not available, cannot gather GPU information")
    except Exception as e:
        logger.warning(f"Error getting GPU information: {str(e)}")
    
    return result


def get_system_summary() -> Dict[str, Any]:
    """
    Get a comprehensive summary of the system environment.
    
    Returns:
        Dict with system information
    """
    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "system": platform.system(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "in_runpod": is_runpod(),
        "gpu": get_gpu_info(),
        "env_vars": {
            "RUNPOD_POD_ID": os.environ.get("RUNPOD_POD_ID"),
            "GPU_NAME": os.environ.get("GPU_NAME"),
            "PROJECT_ROOT": os.environ.get("PROJECT_ROOT"),
            "PYTHONPATH": os.environ.get("PYTHONPATH")
        }
    }
    
    # Get package versions
    try:
        import pip
        package_list = ["torch", "transformers", "pandas", "numpy", "matplotlib", "pillow"]
        summary["packages"] = {}
        
        for package in package_list:
            try:
                __import__(package)
                module = sys.modules[package]
                version = getattr(module, "__version__", "unknown")
                summary["packages"][package] = version
            except (ImportError, KeyError):
                summary["packages"][package] = "not installed"
    except ImportError:
        pass
    
    return summary


def safe_execute(
    func: Callable, 
    error_category: str = "execution",
    error_message: Optional[str] = None,
    recovery_steps: Optional[List[str]] = None,
    default_return: Any = None
) -> Any:
    """
    Safely execute a function with proper error handling.
    
    Args:
        func: Function to execute
        error_category: Category of error if it fails
        error_message: Custom error message (uses function name if None)
        recovery_steps: Suggested steps to recover from failure
        default_return: Value to return if execution fails
        
    Returns:
        Result of the function or default_return on failure
    """
    try:
        return func()
    except Exception as e:
        if error_message is None:
            error_message = f"Error executing {func.__name__}"
        
        # Get traceback information
        tb = traceback.format_exc()
        debug_info = {
            "traceback": tb,
            "system_info": get_system_summary()
        }
        
        error = NotebookFriendlyError(
            message=error_message,
            category=error_category,
            original_error=e,
            recovery_steps=recovery_steps,
            debug_info=debug_info
        )
        
        # Display the error information in a notebook-friendly way
        display_error(error)
        
        return default_return


def display_error(error: Union[NotebookFriendlyError, Exception]) -> None:
    """
    Display an error in a notebook-friendly format.
    
    Args:
        error: Error to display
    """
    if not isinstance(error, NotebookFriendlyError):
        # Convert regular exception to NotebookFriendlyError
        error = NotebookFriendlyError(
            message=str(error),
            original_error=error
        )
    
    # Get the category info
    category_info = ERROR_CATEGORIES.get(error.category, ERROR_CATEGORIES["internal"])
    
    # Format header with timestamps and category
    header = f"{category_info['icon']} {category_info['title']} ({error.timestamp.strftime('%H:%M:%S')})"
    
    # Print error details in a structured format
    print(f"\n{'='*80}")
    print(f"{header}\n{'-'*80}")
    print(f"📌 {error.message}")
    
    if error.original_error and str(error.original_error) != error.message:
        print(f"\n🔍 Original error: {str(error.original_error)}")
    
    if error.recovery_steps:
        print("\n🔄 Recovery steps:")
        for i, step in enumerate(error.recovery_steps, 1):
            print(f"  {i}. {step}")
    
    # Add system info summary for certain categories
    if error.category in ["environment", "gpu", "runpod"]:
        system_info = get_system_summary()
        print("\n💻 System information:")
        print(f"  • Python: {system_info['python_version']}")
        print(f"  • Platform: {system_info['platform']}")
        if system_info['gpu']['available']:
            print(f"  • GPU: {system_info['gpu']['name']} ({system_info['gpu']['memory_total_gb']:.1f} GB)")
            print(f"  • CUDA: {system_info['gpu']['cuda_version']}")
        else:
            print("  • GPU: Not available")
    
    print(f"{'='*80}\n")


def log_error_to_file(error: NotebookFriendlyError, filename: Optional[str] = None) -> str:
    """
    Log detailed error information to a file.
    
    Args:
        error: Error to log
        filename: Custom filename (default: auto-generated based on timestamp)
        
    Returns:
        Path to the log file
    """
    if filename is None:
        timestamp = error.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"error_{error.category}_{timestamp}.log"
    
    # Ensure logs directory exists
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    
    # Full path to log file
    log_path = os.path.join(logs_dir, filename)
    
    with open(log_path, 'w') as f:
        # Write header
        f.write(f"ERROR LOG: {error.category.upper()} - {error.timestamp}\n")
        f.write("="*80 + "\n\n")
        
        # Write error details
        f.write(f"MESSAGE: {error.message}\n\n")
        
        if error.original_error:
            f.write(f"ORIGINAL ERROR: {str(error.original_error)}\n\n")
        
        # Write traceback if available
        if error.debug_info and 'traceback' in error.debug_info:
            f.write("TRACEBACK:\n")
            f.write(error.debug_info['traceback'])
            f.write("\n")
        
        # Write system information
        system_info = error.debug_info.get('system_info', get_system_summary())
        f.write("SYSTEM INFORMATION:\n")
        f.write(f"- Timestamp: {system_info['timestamp']}\n")
        f.write(f"- System: {system_info['system']}\n")
        f.write(f"- Platform: {system_info['platform']}\n")
        f.write(f"- Python Version: {system_info['python_version']}\n")
        f.write(f"- RunPod Environment: {system_info['in_runpod']}\n")
        
        # GPU information
        gpu_info = system_info['gpu']
        f.write("\nGPU INFORMATION:\n")
        f.write(f"- Available: {gpu_info['available']}\n")
        if gpu_info['available']:
            f.write(f"- Name: {gpu_info['name']}\n")
            f.write(f"- Total Memory: {gpu_info['memory_total_gb']:.2f} GB\n")
            f.write(f"- Used Memory: {gpu_info['memory_used_gb']:.2f} GB\n")
            f.write(f"- CUDA Version: {gpu_info['cuda_version']}\n")
            f.write(f"- Driver Version: {gpu_info['driver_version']}\n")
        
        # Package versions
        if 'packages' in system_info:
            f.write("\nPACKAGE VERSIONS:\n")
            for package, version in system_info['packages'].items():
                f.write(f"- {package}: {version}\n")
        
        # Environment variables
        f.write("\nENVIRONMENT VARIABLES:\n")
        for var, value in system_info['env_vars'].items():
            f.write(f"- {var}: {value}\n")
        
        # Recovery steps
        if error.recovery_steps:
            f.write("\nRECOVERY STEPS:\n")
            for i, step in enumerate(error.recovery_steps, 1):
                f.write(f"{i}. {step}\n")
    
    logger.info(f"Error details logged to {log_path}")
    return log_path


def create_error_report(errors: List[NotebookFriendlyError]) -> str:
    """
    Create a comprehensive error report from multiple errors.
    
    Args:
        errors: List of errors to include in the report
        
    Returns:
        Path to the generated report file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"error_report_{timestamp}.html"
    
    # Ensure logs directory exists
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    
    # Full path to report file
    report_path = os.path.join(logs_dir, filename)
    
    # Basic HTML template
    html = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "    <title>Notebook Error Report</title>",
        "    <style>",
        "        body { font-family: Arial, sans-serif; margin: 20px; }",
        "        .error { border: 1px solid #ddd; padding: 15px; margin-bottom: 20px; border-radius: 5px; }",
        "        .error-header { background-color: #f8f8f8; padding: 10px; margin: -15px -15px 15px -15px; border-bottom: 1px solid #ddd; }",
        "        .environment { background-color: #e6f7ff; }",
        "        .model { background-color: #f0f0ff; }",
        "        .data { background-color: #fff8e6; }",
        "        .config { background-color: #f0fff0; }",
        "        .execution { background-color: #fff0f0; }",
        "        .gpu { background-color: #f5e6ff; }",
        "        .runpod { background-color: #e6fffa; }",
        "        .internal { background-color: #fff0e6; }",
        "        .timestamp { color: #999; font-size: 0.9em; }",
        "        .message { font-weight: bold; margin: 10px 0; }",
        "        .original-error { font-family: monospace; background-color: #f8f8f8; padding: 10px; margin: 10px 0; }",
        "        .recovery { background-color: #f8f8f8; padding: 10px; margin: 10px 0; }",
        "        h1, h2 { color: #333; }",
        "        .summary { margin-bottom: 30px; }",
        "    </style>",
        "</head>",
        "<body>",
        f"    <h1>Notebook Error Report</h1>",
        f"    <p class='timestamp'>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
        "    <div class='summary'>",
        f"        <h2>Summary</h2>",
        f"        <p>Total errors: {len(errors)}</p>",
    ]
    
    # Add category counts to summary
    category_counts = {}
    for error in errors:
        category = error.category
        if category not in category_counts:
            category_counts[category] = 0
        category_counts[category] += 1
    
    html.append("        <ul>")
    for category, count in category_counts.items():
        category_info = ERROR_CATEGORIES.get(category, ERROR_CATEGORIES["internal"])
        html.append(f"            <li>{category_info['icon']} {category_info['title']}: {count}</li>")
    html.append("        </ul>")
    html.append("    </div>")
    
    html.append("    <h2>Detailed Errors</h2>")
    
    # Add each error
    for i, error in enumerate(errors, 1):
        category_info = ERROR_CATEGORIES.get(error.category, ERROR_CATEGORIES["internal"])
        
        html.extend([
            f"    <div class='error {error.category}'>",
            f"        <div class='error-header'>",
            f"            <h3>{category_info['icon']} {category_info['title']} (Error #{i})</h3>",
            f"            <p class='timestamp'>Occurred at {error.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>",
            f"        </div>",
            f"        <p class='message'>{error.message}</p>"
        ])
        
        # Original error
        if error.original_error and str(error.original_error) != error.message:
            html.append(f"        <div class='original-error'>")
            html.append(f"            <p><strong>Original error:</strong> {str(error.original_error)}</p>")
            html.append(f"        </div>")
        
        # Recovery steps
        if error.recovery_steps:
            html.append(f"        <div class='recovery'>")
            html.append(f"            <p><strong>Recovery steps:</strong></p>")
            html.append(f"            <ol>")
            for step in error.recovery_steps:
                html.append(f"                <li>{step}</li>")
            html.append(f"            </ol>")
            html.append(f"        </div>")
        
        html.append("    </div>")
    
    # Close HTML document
    html.extend([
        "</body>",
        "</html>"
    ])
    
    # Write to file
    with open(report_path, 'w') as f:
        f.write('\n'.join(html))
    
    logger.info(f"Error report generated at {report_path}")
    return report_path


# If executed directly, perform self-tests
if __name__ == "__main__":
    # Demonstrate error handling
    print("Testing error handling utilities...")
    
    # Create example errors
    errors = []
    
    # Environment error
    env_error = NotebookFriendlyError(
        message="Could not detect GPU properly",
        category="environment",
        original_error=RuntimeError("CUDA initialization failed"),
        recovery_steps=[
            "Restart the kernel",
            "Verify CUDA is properly installed",
            "Check nvidia-smi output"
        ]
    )
    display_error(env_error)
    errors.append(env_error)
    
    # Model error
    model_error = NotebookFriendlyError(
        message="Failed to load model weights",
        category="model",
        original_error=FileNotFoundError("No such file or directory: 'model/weights.bin'"),
        recovery_steps=[
            "Check model path",
            "Download model files if missing",
            "Verify model compatibility with current environment"
        ]
    )
    errors.append(model_error)
    
    # Create and save error report
    report_path = create_error_report(errors)
    print(f"Error report generated: {report_path}")
    
    # Test safe execution function
    def working_function():
        return "Success!"
    
    def failing_function():
        raise ValueError("This function intentionally fails")
    
    print("\nTesting safe_execute with successful function:")
    result = safe_execute(working_function)
    print(f"Result: {result}")
    
    print("\nTesting safe_execute with failing function:")
    result = safe_execute(
        failing_function,
        error_category="execution",
        error_message="Example of handling a failing function",
        recovery_steps=["Fix the function", "Try again with different parameters"],
        default_return="Default result"
    )
    print(f"Result: {result}")
    
    print("\nAll tests completed!") 