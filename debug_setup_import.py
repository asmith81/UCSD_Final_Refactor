#!/usr/bin/env python3
"""
Diagnostic script to trace import issues with setup_utils.py
"""

import os
import sys
import importlib
import traceback

# Add the project root to the path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

def trace_import(module_name, max_depth=10):
    """Trace an import to find where it fails"""
    print(f"\nTracing import of: {module_name}")
    
    parts = module_name.split('.')
    current = ""
    
    for i, part in enumerate(parts):
        if current:
            current = f"{current}.{part}"
        else:
            current = part
            
        try:
            module = importlib.import_module(current)
            print(f"✅ Successfully imported: {current}")
            
            if i == len(parts) - 1:
                # Print some info about the final module
                print(f"\nModule details for {current}:")
                print(f"File: {module.__file__}")
                print(f"Directory: {os.path.dirname(module.__file__)}")
                
                # Print all exported symbols
                symbols = dir(module)
                print(f"\nExported symbols ({len(symbols)}):")
                for sym in sorted(symbols):
                    if not sym.startswith('_'):
                        print(f"  - {sym}")
        except Exception as e:
            print(f"❌ Failed to import: {current}")
            print(f"Error: {str(e)}")
            traceback.print_exc()
            break

def check_specific_imports():
    """Check specific imports that might be causing issues"""
    print("\n--- Checking specific imports ---")
    
    try:
        from src.prompts.registry import Prompt
        print("✅ Successfully imported Prompt from src.prompts.registry")
    except Exception as e:
        print(f"❌ Failed to import Prompt from src.prompts.registry")
        print(f"Error: {str(e)}")
        traceback.print_exc()
    
    try:
        from src.prompts import Prompt
        print("✅ Successfully imported Prompt from src.prompts")
    except Exception as e:
        print(f"❌ Failed to import Prompt from src.prompts")
        print(f"Error: {str(e)}")
        traceback.print_exc()
    
    try:
        from src.notebook.setup_utils import get_system_info
        print("✅ Successfully imported get_system_info from src.notebook.setup_utils")
    except Exception as e:
        print(f"❌ Failed to import get_system_info from src.notebook.setup_utils")
        print(f"Error: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    # Trace the main imports
    trace_import("src.notebook.setup_utils")
    trace_import("src.prompts.registry")
    
    # Check specific imports
    check_specific_imports()
    
    print("\nDiagnostic complete.") 