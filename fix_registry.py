#!/usr/bin/env python3
"""
Script to verify and fix the prompt registry.
This will check if the get_prompt_registry function exists in registry.py and add it if missing.
"""

import os
import sys
import re

def fix_registry_file(file_path):
    print(f"Checking {file_path}...")
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist!")
        return False
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if get_prompt_registry function already exists
    if 'def get_prompt_registry(' in content:
        print("The get_prompt_registry function already exists in the file.")
        return True
    
    # Check if get_registry function exists (we need to add our alias after this)
    get_registry_match = re.search(r'def get_registry\(\).*?return.*?_registry', content, re.DOTALL)
    if not get_registry_match:
        print("Error: Could not find the get_registry function to add alias!")
        return False
    
    get_registry_block = get_registry_match.group(0)
    position = content.find(get_registry_block) + len(get_registry_block)
    
    # Create the get_prompt_registry function
    prompt_registry_fn = '''

def get_prompt_registry() -> PromptRegistry:
    """Get the global prompt registry instance (alias for get_registry)."""
    return get_registry()
'''
    
    # Insert the new function after get_registry
    new_content = content[:position] + prompt_registry_fn + content[position:]
    
    # Write back to file
    with open(file_path, 'w') as f:
        f.write(new_content)
    
    print("✅ Added get_prompt_registry function to the file!")
    return True

def main():
    # Default path in the RunPod environment
    registry_path = '/workspace/src/prompts/registry.py'
    
    # Allow overriding the path with a command-line argument
    if len(sys.argv) > 1:
        registry_path = sys.argv[1]
    
    # Fix the registry file
    if fix_registry_file(registry_path):
        print("Fix completed successfully!")
    else:
        print("Failed to fix the registry file.")

if __name__ == "__main__":
    main() 