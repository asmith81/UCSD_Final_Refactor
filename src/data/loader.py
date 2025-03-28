"""
Data Loading Utilities

This module provides functions for loading and processing data files,
particularly focused on loading ground truth data and mapping it to
corresponding images for the invoice processing system.
"""

import os
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

# Set up logging
logger = logging.getLogger(__name__)

def load_ground_truth(
    file_path: Union[str, Path], 
    id_column: Optional[str] = None
) -> pd.DataFrame:
    """
    Load ground truth data from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        id_column: Optional name of the ID column
        
    Returns:
        DataFrame containing the ground truth data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file isn't valid CSV
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        logger.error(f"Ground truth file not found: {file_path}")
        raise FileNotFoundError(f"Ground truth file not found: {file_path}")
    
    try:
        # Load the CSV file
        df = pd.read_csv(file_path)
        logger.info(f"Loaded ground truth data with {len(df)} records")
        
        # If ID column is specified, ensure it exists
        if id_column and id_column not in df.columns:
            logger.warning(f"ID column '{id_column}' not found in ground truth data")
            logger.info(f"Available columns: {', '.join(df.columns)}")
            
            # Try to find a suitable column if the specified one doesn't exist
            candidates = ["invoice", "image", "filename", "file", "id"]
            for candidate in candidates:
                matches = [col for col in df.columns if candidate.lower() in col.lower()]
                if matches:
                    id_column = matches[0]
                    logger.info(f"Using '{id_column}' as ID column instead")
                    break
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading ground truth data: {e}")
        raise

def map_images_to_ground_truth(
    ground_truth_df: pd.DataFrame,
    image_paths: List[Path],
    id_column: str,
    field_columns: Dict[str, str]
) -> Dict[str, Dict[str, Any]]:
    """
    Map images to their corresponding ground truth values.
    
    Args:
        ground_truth_df: DataFrame with ground truth data
        image_paths: List of paths to image files
        id_column: Name of the column containing image identifiers
        field_columns: Dictionary mapping field types to column names
        
    Returns:
        Dictionary mapping image IDs to their ground truth values
    """
    if not isinstance(ground_truth_df, pd.DataFrame):
        logger.error("Invalid ground truth data provided")
        raise ValueError("Ground truth must be a pandas DataFrame")
    
    if id_column not in ground_truth_df.columns:
        logger.error(f"ID column '{id_column}' not found in ground truth data")
        raise ValueError(f"ID column '{id_column}' not found in ground truth data")
    
    # Validate field columns
    for field_type, column_name in field_columns.items():
        if column_name not in ground_truth_df.columns:
            logger.error(f"Field column '{column_name}' for '{field_type}' not found in ground truth data")
            raise ValueError(f"Field column '{column_name}' not found in ground truth data")
    
    # Create mapping
    mapping = {}
    unmatched_images = []
    
    for image_path in image_paths:
        image_id = image_path.stem  # Get filename without extension
        
        # Find matching row in ground truth
        # Converting to string to ensure matching works with numeric IDs
        matching_rows = ground_truth_df[ground_truth_df[id_column].astype(str) == image_id]
        
        if len(matching_rows) > 0:
            row = matching_rows.iloc[0]
            
            # Create entry with image path and all requested fields
            entry = {
                'image_path': str(image_path),
                'image_id': image_id
            }
            
            # Add ground truth values for each field
            for field_type, column_name in field_columns.items():
                entry[field_type] = str(row[column_name]).strip()
            
            mapping[image_id] = entry
        else:
            unmatched_images.append(image_id)
            logger.warning(f"No matching ground truth found for image {image_id}")
    
    logger.info(f"Successfully mapped {len(mapping)} images to ground truth data")
    
    if unmatched_images:
        logger.warning(f"Found {len(unmatched_images)} images without ground truth data")
        if len(unmatched_images) < 10:
            logger.warning(f"Unmatched images: {', '.join(unmatched_images)}")
        else:
            logger.warning(f"First 10 unmatched images: {', '.join(unmatched_images[:10])}...")
    
    return mapping

def get_image_files(
    image_dir: Union[str, Path],
    extensions: List[str] = None
) -> List[Path]:
    """
    Get all image files in a directory.
    
    Args:
        image_dir: Directory to search for images
        extensions: List of file extensions to include (default: ['.jpg', '.png'])
        
    Returns:
        List of image file paths
    """
    if extensions is None:
        extensions = ['.jpg', '.png']
    
    image_dir = Path(image_dir)
    
    if not image_dir.exists():
        logger.error(f"Image directory not found: {image_dir}")
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    
    image_files = []
    
    for ext in extensions:
        if not ext.startswith('.'):
            ext = '.' + ext
        image_files.extend(list(image_dir.glob(f"*{ext}")))
    
    logger.info(f"Found {len(image_files)} image files in {image_dir}")
    return image_files

def prepare_batch_items(
    ground_truth_mapping: Dict[str, Dict[str, Any]],
    field_type: str
) -> List[Dict[str, Any]]:
    """
    Create structured batch items for processing.
    
    Args:
        ground_truth_mapping: Mapping from image IDs to ground truth data
        field_type: The field type to extract
        
    Returns:
        List of batch items ready for processing
    """
    batch_items = []
    
    for image_id, data in ground_truth_mapping.items():
        # Skip if the requested field is not in the mapping
        if field_type not in data:
            logger.warning(f"Field '{field_type}' not found for image {image_id}, skipping")
            continue
        
        batch_items.append({
            "image_id": image_id,
            "image_path": data["image_path"],
            "ground_truth": data[field_type],
            "field_type": field_type
        })
    
    logger.info(f"Created {len(batch_items)} batch items for field: {field_type}")
    return batch_items

def load_and_prepare_data(
    ground_truth_path: Union[str, Path],
    image_dir: Union[str, Path],
    field_to_extract: str,
    field_column_name: str,
    image_id_column: str = "Invoice"
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Convenience function to load ground truth, map images, and prepare batch items.
    
    Args:
        ground_truth_path: Path to ground truth CSV
        image_dir: Directory containing images
        field_to_extract: Field to extract from invoices
        field_column_name: Column in CSV containing the field data
        image_id_column: Column in CSV with image identifiers
        
    Returns:
        Tuple of (ground_truth_df, ground_truth_mapping, batch_items)
    """
    # Load ground truth data
    ground_truth_df = load_ground_truth(ground_truth_path, id_column=image_id_column)
    
    # Get image files
    image_files = get_image_files(image_dir)
    
    # Map images to ground truth
    ground_truth_mapping = map_images_to_ground_truth(
        ground_truth_df,
        image_files,
        id_column=image_id_column,
        field_columns={field_to_extract: field_column_name}
    )
    
    # Prepare batch items
    batch_items = prepare_batch_items(ground_truth_mapping, field_to_extract)
    
    return ground_truth_df, ground_truth_mapping, batch_items