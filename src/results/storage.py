"""
Results Storage Management

This module provides a comprehensive system for storing, retrieving, 
and managing experimental results across different storage backends.
"""

import os
import json
import yaml
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union

class ResultStorage:
    """
    Flexible result storage management with multiple backend support.
    
    Supports:
    - Local filesystem storage
    - Versioning
    - Metadata tracking
    - Multiple file format support
    """
    
    def __init__(
        self, 
        base_path: Union[str, Path], 
        experiment_name: Optional[str] = None
    ):
        """
        Initialize result storage system.
        
        Args:
            base_path: Root directory for storing results
            experiment_name: Optional name for current experiment
        """
        self.base_path = Path(base_path)
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure base directories exist
        self._setup_directories()
    
    def _setup_directories(self):
        """Create necessary directory structure for result storage."""
        paths = {
            'raw': self.base_path / 'raw',
            'processed': self.base_path / 'processed',
            'metadata': self.base_path / 'metadata',
            'archives': self.base_path / 'archives'
        }
        
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
    
    def _generate_filename(
        self, 
        prefix: str = '', 
        field: Optional[str] = None, 
        extension: str = 'json'
    ) -> str:
        """
        Generate a unique, timestamped filename.
        
        Args:
            prefix: Optional prefix for filename
            field: Optional field name
            extension: File extension
        
        Returns:
            Unique filename
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        components = [
            prefix,
            field or '',
            timestamp,
            self._generate_hash()
        ]
        filename = '_'.join(filter(bool, components))
        return f"{filename}.{extension}"
    
    def _generate_hash(self, length: int = 6) -> str:
        """
        Generate a short unique hash.
        
        Args:
            length: Length of hash
        
        Returns:
            Short hash string
        """
        return hashlib.md5(
            str(datetime.now().timestamp()).encode()
        ).hexdigest()[:length]
    
    def save_results(
        self, 
        results: Dict[str, Any], 
        field: Optional[str] = None,
        format: str = 'json'
    ) -> Path:
        """
        Save experiment results to storage.
        
        Args:
            results: Results dictionary to save
            field: Optional field name for categorization
            format: Storage format (json or yaml)
        
        Returns:
            Path to saved file
        """
        filename = self._generate_filename(
            prefix='results', 
            field=field, 
            extension=format
        )
        
        # Determine storage path based on format
        storage_path = (
            self.base_path / 'raw' / filename 
            if format == 'json' else 
            self.base_path / 'processed' / filename
        )
        
        # Save results
        with open(storage_path, 'w') as f:
            if format == 'json':
                json.dump(results, f, indent=2)
            elif format == 'yaml':
                yaml.safe_dump(results, f, default_flow_style=False)
        
        # Create metadata entry
        self._create_metadata_entry(storage_path, results)
        
        return storage_path
    
    def _create_metadata_entry(
        self, 
        file_path: Path, 
        results: Dict[str, Any]
    ):
        """
        Create a metadata entry for stored results.
        
        Args:
            file_path: Path to stored results file
            results: Results dictionary
        """
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'filename': file_path.name,
            'experiment': self.experiment_name,
            'size_bytes': file_path.stat().st_size,
            'fields': list(results.keys()) if isinstance(results, dict) else []
        }
        
        metadata_path = (
            self.base_path / 'metadata' / 
            f"{file_path.stem}_metadata.json"
        )
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_results(
        self, 
        filename: Optional[str] = None,
        field: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load results from storage.
        
        Args:
            filename: Specific filename to load
            field: Optional field filter
        
        Returns:
            Loaded results
        """
        # If no filename provided, find most recent
        if not filename:
            search_path = self.base_path / 'raw'
            files = sorted(
                search_path.glob('results*.json'), 
                key=os.path.getmtime, 
                reverse=True
            )
            filename = files[0].name if files else None
        
        if not filename:
            raise FileNotFoundError("No results files found")
        
        file_path = self.base_path / 'raw' / filename
        
        with open(file_path, 'r') as f:
            results = json.load(f)
        
        # Optional field filtering
        if field and isinstance(results, dict):
            results = {k: v for k, v in results.items() if k == field}
        
        return results