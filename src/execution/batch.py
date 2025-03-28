"""
Batch processing utilities for invoice image extraction.

This module provides functionality for:
1. Creating and managing batches of images for efficient processing
2. Processing batches with automatic memory management
3. Tracking progress and providing status updates
4. Implementing checkpointing for resumable operations
5. Optimizing batch sizes based on available resources
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from datetime import datetime

import torch
import numpy as np
from tqdm.auto import tqdm
from IPython.display import clear_output, display
import matplotlib.pyplot as plt

# Import project modules
from src.execution.inference import process_image_with_metrics, extract_field_from_image
from src.models.loader import load_model_and_processor, optimize_model_memory, get_gpu_memory_info
from src.config.environment import get_environment_config

# Set up logging
logger = logging.getLogger(__name__)


def estimate_optimal_batch_size(
    model_name: str,
    image_path: Union[str, Path],
    prompt: Any,
    max_batch_size: int = 16,
    memory_threshold: float = 0.8,
    **kwargs
) -> int:
    """
    Estimate optimal batch size based on GPU memory constraints.
    
    Args:
        model_name: Name of the model to use
        image_path: Path to a sample image for testing
        prompt: Prompt to use for testing
        max_batch_size: Maximum batch size to consider
        memory_threshold: Maximum fraction of GPU memory to use
        **kwargs: Additional parameters for extraction
        
    Returns:
        Optimal batch size (1 if estimation fails)
    """
    if not torch.cuda.is_available():
        logger.info("No GPU available, defaulting to batch size 1")
        return 1
    
    # Get total GPU memory
    gpu_info = get_gpu_memory_info()
    total_memory = gpu_info.get("total_memory_gb", 0)
    threshold_memory = total_memory * memory_threshold
    
    logger.info(f"Estimating optimal batch size for {model_name} on {gpu_info.get('device_name', 'GPU')}")
    logger.info(f"Total GPU memory: {total_memory:.2f} GB, threshold: {threshold_memory:.2f} GB")
    
    # Load model for testing
    try:
        model, processor = load_model_and_processor(model_name)
        
        # Start with a single image and measure memory usage
        _, _, metadata = extract_field_from_image(
            image_path=image_path,
            prompt=prompt,
            model_name=model_name,
            model=model,
            processor=processor,
            **kwargs
        )
        
        # Get memory used for a single image
        if "inference_memory_delta_gb" in metadata:
            single_image_memory = metadata["inference_memory_delta_gb"]
            current_memory = metadata.get("post_inference_memory_gb", 0)
            
            logger.info(f"Memory for single image: {single_image_memory:.4f} GB")
            logger.info(f"Current GPU memory usage: {current_memory:.4f} GB")
            
            # Calculate available memory and optimal batch size
            available_memory = threshold_memory - current_memory + single_image_memory
            if available_memory <= 0:
                logger.warning("Not enough available memory for batching, using batch size 1")
                return 1
                
            optimal_batch = min(max_batch_size, max(1, int(available_memory / single_image_memory)))
            
            logger.info(f"Estimated optimal batch size: {optimal_batch}")
            return optimal_batch
        else:
            logger.warning("Memory delta information not available, defaulting to batch size 1")
            return 1
    
    except Exception as e:
        logger.error(f"Error estimating batch size: {e}")
        logger.info("Defaulting to conservative batch size 1")
        return 1
    
    finally:
        # Clean up resources
        optimize_memory(clear_cache=True)


def create_batches(
    items: List[Any],
    batch_size: int = 1
) -> List[List[Any]]:
    """
    Split a list of items into batches of specified size.
    
    Args:
        items: List of items to batch
        batch_size: Size of each batch
        
    Returns:
        List of batches, where each batch is a list of items
    """
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def save_checkpoint(
    results: List[Dict[str, Any]],
    pending_items: List[Dict[str, Any]],
    checkpoint_path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save processing state to a checkpoint file for later resumption.
    
    Args:
        results: List of processed results
        pending_items: List of items not yet processed
        checkpoint_path: Path to save the checkpoint
        metadata: Optional metadata about the processing state
    """
    checkpoint_data = {
        "timestamp": datetime.now().isoformat(),
        "results": results,
        "pending_items": pending_items,
        "metadata": metadata or {}
    }
    
    # Ensure directory exists
    checkpoint_dir = str(Path(checkpoint_path).parent)
    if checkpoint_dir and not Path(checkpoint_dir).exists():
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Save checkpoint
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f)
        
    logger.info(f"Checkpoint saved to {checkpoint_path}")
    logger.info(f"Processed: {len(results)}, Pending: {len(pending_items)}")


def load_checkpoint(checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load processing state from a checkpoint file.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        Dictionary with checkpoint data
        
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        ValueError: If checkpoint data is invalid
    """
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    with open(checkpoint_path, 'r') as f:
        checkpoint_data = json.load(f)
    
    # Validate checkpoint structure
    if not all(k in checkpoint_data for k in ["results", "pending_items"]):
        raise ValueError("Invalid checkpoint data structure")
    
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    logger.info(f"Checkpoint contains {len(checkpoint_data['results'])} processed items "
               f"and {len(checkpoint_data['pending_items'])} pending items")
    
    return checkpoint_data


def process_batch(
    batch_items: List[Dict[str, Any]],
    model_name: str,
    prompt: Any,
    field_type: str,
    model: Any = None,
    processor: Any = None,
    extraction_rules: Optional[Dict[str, Any]] = None,
    metrics: List[str] = ["exact_match", "character_error_rate"],
    show_progress: bool = False,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Process a batch of images with the same model and prompt.
    
    Args:
        batch_items: List of dictionaries with image_path and ground_truth
        model_name: Name of the model to use
        prompt: Prompt to use for extraction
        field_type: Type of field to extract
        model: Pre-loaded model (will be loaded if None)
        processor: Pre-loaded processor (will be loaded if None)
        extraction_rules: Optional rules for field extraction
        metrics: List of metrics to calculate
        show_progress: Whether to show progress bar for this batch
        **kwargs: Additional parameters for extraction
        
    Returns:
        List of results for each item in the batch
    """
    results = []
    
    # Load model if not provided
    if model is None or processor is None:
        logger.info(f"Loading model {model_name} for batch processing")
        model, processor = load_model_and_processor(model_name, **kwargs)
    
    # Create progress iterator if requested
    items_iter = tqdm(batch_items) if show_progress else batch_items
    
    # Process each item in the batch
    for item in items_iter:
        try:
            # Get required parameters
            image_path = item["image_path"]
            ground_truth = item["ground_truth"]
            
            # Process image
            result = process_image_with_metrics(
                image_path=image_path,
                ground_truth=ground_truth,
                prompt=prompt,
                model_name=model_name,
                field_type=field_type,
                model=model,
                processor=processor,
                extraction_rules=extraction_rules,
                metrics=metrics,
                **kwargs
            )
            
            # Add to results
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing item {item.get('image_id', 'unknown')}: {e}")
            
            # Add error result
            error_result = {
                "image_id": item.get("image_id", Path(item["image_path"]).stem),
                "ground_truth": item["ground_truth"],
                "raw_extraction": "ERROR",
                "processed_extraction": "ERROR",
                "field_type": field_type,
                "processing_time": 0.0,
                "model_name": model_name,
                "exact_match": False,
                "character_error_rate": 1.0,
                "error": str(e)
            }
            results.append(error_result)
    
    return results


def format_time(seconds):
    """Format seconds into a readable string with hours, minutes, seconds."""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    elif minutes > 0:
        return f"{int(minutes)}m {int(seconds)}s"
    else:
        return f"{int(seconds)}s"


def display_progress_visualization(
    completed: int,
    total: int,
    successful: int,
    failed: int,
    avg_time_per_item: float,
    elapsed_time: float,
    batch_idx: int = 0,
    n_batches: int = 0,
    recent_times: List[float] = None,
    recent_errors: List[Dict[str, Any]] = None
):
    """
    Display a visualization of extraction progress.
    
    Args:
        completed: Number of completed items
        total: Total number of items
        successful: Number of successful extractions
        failed: Number of failed extractions
        avg_time_per_item: Average processing time per item
        elapsed_time: Total elapsed time so far
        batch_idx: Current batch index
        n_batches: Total number of batches
        recent_times: List of recent processing times
        recent_errors: List of recent error information
    """
    # Clear previous output
    clear_output(wait=True)
    
    # Calculate metrics
    remaining = total - completed
    success_rate = (successful / completed) * 100 if completed > 0 else 0
    eta_seconds = avg_time_per_item * remaining if avg_time_per_item > 0 else 0
    
    # Create figure with 2x2 grid
    plt.figure(figsize=(14, 10))
    
    # Plot 1: Overall Progress
    plt.subplot(2, 2, 1)
    plt.bar(['Completed', 'Remaining'], [completed, remaining], color=['#3498db', '#e74c3c'])
    plt.title(f'Overall Progress: {completed}/{total} images ({completed/total*100:.1f}%)')
    plt.text(0, completed/2, f"{completed}", ha='center', va='center', color='white', fontweight='bold')
    if remaining > 0:
        plt.text(1, remaining/2, f"{remaining}", ha='center', va='center', color='white', fontweight='bold')
    
    # Plot 2: Success vs Failure
    plt.subplot(2, 2, 2)
    plt.bar(['Success', 'Failure'], [successful, failed], color=['#2ecc71', '#e74c3c'])
    plt.title(f'Extraction Results: {success_rate:.1f}% Success Rate')
    if successful > 0:
        plt.text(0, successful/2, f"{successful}", ha='center', va='center', color='white', fontweight='bold')
    if failed > 0:
        plt.text(1, failed/2, f"{failed}", ha='center', va='center', color='white', fontweight='bold')
    
    # Plot 3: Processing Time Trend (if data available)
    plt.subplot(2, 2, 3)
    if recent_times and len(recent_times) > 1:
        plt.plot(range(len(recent_times)), recent_times, 'o-', color='#9b59b6')
        plt.axhline(y=avg_time_per_item, color='#e74c3c', linestyle='--', label=f'Average: {avg_time_per_item:.2f}s')
        plt.title(f'Processing Time Trend (recent {len(recent_times)} items)')
        plt.xlabel('Items')
        plt.ylabel('Seconds per item')
        plt.legend()
    else:
        plt.text(0.5, 0.5, f"Average processing time: {avg_time_per_item:.2f}s per item", 
                 ha='center', va='center', fontsize=12)
        plt.title('Processing Time')
        plt.axis('off')
    
    # Plot 4: Recent Errors (if any)
    plt.subplot(2, 2, 4)
    if recent_errors and len(recent_errors) > 0:
        plt.axis('off')
        plt.title(f'Recent Errors ({len(recent_errors)} total)')
        
        error_text = "\n".join([
            f"{i+1}. Image: {err.get('image_id', 'unknown')} - {err.get('error', 'Unknown error')[:50]}..."
            for i, err in enumerate(recent_errors[:3])
        ])
        
        plt.text(0.1, 0.5, error_text, va='center', fontsize=10, wrap=True)
    else:
        plt.text(0.5, 0.5, "No recent errors", ha='center', va='center', fontsize=12)
        plt.title('Errors')
        plt.axis('off')
    
    # Add overall status information
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Add title with timing information
    status_text = (
        f"Extraction Progress - Batch {batch_idx+1}/{n_batches} | "
        f"Elapsed: {format_time(elapsed_time)} | "
        f"ETA: {format_time(eta_seconds)}"
    )
    plt.suptitle(status_text, fontsize=16)
    
    # Show the plot
    plt.show()
    
    # Print summary text as well (for non-graphical environments)
    print(f"Progress: {completed}/{total} images ({completed/total*100:.1f}%)")
    print(f"Success Rate: {success_rate:.1f}% ({successful} successful, {failed} failed)")
    print(f"Timing: {avg_time_per_item:.2f}s per image | Elapsed: {format_time(elapsed_time)} | ETA: {format_time(eta_seconds)}")
    print(f"Batch Progress: {batch_idx+1}/{n_batches} batches")


def process_batches(
    items: List[Dict[str, Any]],
    model_name: str,
    prompt: Any,
    field_type: str,
    batch_size: int = 1,
    checkpoint_path: Optional[Union[str, Path]] = None,
    checkpoint_frequency: int = 5,
    resume_from_checkpoint: bool = True,
    extraction_rules: Optional[Dict[str, Any]] = None,
    metrics: List[str] = ["exact_match", "character_error_rate"],
    show_progress: bool = True,
    optimize_after_batch: bool = True,
    show_visualization: bool = True,
    visualization_frequency: int = 1,  # Update visualization every N batches
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Process a list of items in batches with checkpointing and visual progress tracking.
    
    Args:
        items: List of dictionaries with image_path and ground_truth
        model_name: Name of the model to use
        prompt: Prompt to use for extraction
        field_type: Type of field to extract
        batch_size: Number of items to process in each batch
        checkpoint_path: Path to save checkpoints (None for no checkpointing)
        checkpoint_frequency: How many batches to process between checkpoints
        resume_from_checkpoint: Whether to resume from existing checkpoint
        extraction_rules: Optional rules for field extraction
        metrics: List of metrics to calculate
        show_progress: Whether to show overall progress bar
        optimize_after_batch: Whether to run memory optimization after each batch
        show_visualization: Whether to show visualization of progress
        visualization_frequency: How often to update the visualization (in batches)
        **kwargs: Additional parameters for extraction
        
    Returns:
        List of results for all processed items
    """
    results = []
    start_time = time.time()
    
    # Statistics for visualization
    successful_extractions = 0
    failed_extractions = 0
    processing_times = []
    recent_errors = []
    
    # Try to resume from checkpoint if requested
    if checkpoint_path and resume_from_checkpoint and Path(checkpoint_path).exists():
        try:
            logger.info(f"Attempting to resume from checkpoint: {checkpoint_path}")
            checkpoint_data = load_checkpoint(checkpoint_path)
            results = checkpoint_data["results"]
            items_to_process = checkpoint_data["pending_items"]
            
            # Update counts for visualization
            successful_extractions = sum(1 for r in results if r.get("exact_match", False))
            failed_extractions = len(results) - successful_extractions
            
            logger.info(f"Resuming with {len(results)} existing results, {len(items_to_process)} items remaining")
        except Exception as e:
            logger.error(f"Error resuming from checkpoint: {e}")
            logger.info("Starting from beginning")
            items_to_process = items
    else:
        items_to_process = items
    
    # Return early if there's nothing to process
    if not items_to_process:
        logger.info("No items to process")
        return results
    
    # Create batches
    batches = create_batches(items_to_process, batch_size)
    n_batches = len(batches)
    total_items = len(items)
    completed_items = len(results)
    
    logger.info(f"Processing {len(items_to_process)} items in {n_batches} batches of size {batch_size}")
    
    # Show initial visualization if requested
    if show_visualization and show_progress:
        display_progress_visualization(
            completed=completed_items,
            total=total_items,
            successful=successful_extractions,
            failed=failed_extractions,
            avg_time_per_item=0.0,
            elapsed_time=0.0,
            batch_idx=0,
            n_batches=n_batches
        )
    
    # Load model once for all batches
    logger.info(f"Loading model {model_name}")
    model, processor = load_model_and_processor(model_name, **kwargs)
    
    # Create progress bar for all batches if requested
    batch_iter = tqdm(enumerate(batches), total=n_batches, desc="Processing batches") if show_progress else enumerate(batches)
    
    try:
        # Process each batch
        for batch_idx, batch in batch_iter:
            batch_start = time.time()
            logger.info(f"Processing batch {batch_idx+1}/{n_batches} ({len(batch)} items)")
            
            # Process the batch
            batch_results = process_batch(
                batch_items=batch,
                model_name=model_name,
                prompt=prompt,
                field_type=field_type,
                model=model,
                processor=processor,
                extraction_rules=extraction_rules,
                metrics=metrics,
                show_progress=False,  # Don't show nested progress bars
                **kwargs
            )
            
            # Update statistics
            batch_successful = sum(1 for r in batch_results if r.get("exact_match", False))
            batch_failed = len(batch_results) - batch_successful
            
            successful_extractions += batch_successful
            failed_extractions += batch_failed
            
            # Collect recent processing times
            batch_times = [r.get("processing_time", 0.0) for r in batch_results if "error" not in r]
            if batch_times:
                processing_times.extend(batch_times)
                # Keep only the most recent 100 times for visualization
                if len(processing_times) > 100:
                    processing_times = processing_times[-100:]
            
            # Collect recent errors
            batch_errors = [r for r in batch_results if "error" in r]
            if batch_errors:
                recent_errors.extend(batch_errors)
                # Keep only the most recent 10 errors
                if len(recent_errors) > 10:
                    recent_errors = recent_errors[-10:]
            
            # Add batch results to overall results
            results.extend(batch_results)
            completed_items += len(batch)
            
            # Calculate statistics for visualization
            elapsed_time = time.time() - start_time
            avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
            
            # Log batch completion
            batch_time = time.time() - batch_start
            logger.info(f"Batch {batch_idx+1} completed in {batch_time:.2f}s "
                       f"({batch_time/len(batch):.2f}s per item)")
            
            # Update visualization if requested
            if show_visualization and show_progress and (batch_idx % visualization_frequency == 0 or batch_idx == n_batches - 1):
                display_progress_visualization(
                    completed=completed_items,
                    total=total_items,
                    successful=successful_extractions,
                    failed=failed_extractions,
                    avg_time_per_item=avg_time,
                    elapsed_time=elapsed_time,
                    batch_idx=batch_idx,
                    n_batches=n_batches,
                    recent_times=processing_times[-20:] if len(processing_times) > 20 else processing_times,
                    recent_errors=recent_errors
                )
            
            # Save checkpoint if requested
            if checkpoint_path and (batch_idx + 1) % checkpoint_frequency == 0:
                remaining_items = [item for batch in batches[batch_idx+1:] for item in batch]
                save_checkpoint(
                    results=results,
                    pending_items=remaining_items,
                    checkpoint_path=checkpoint_path,
                    metadata={
                        "model_name": model_name,
                        "field_type": field_type,
                        "total_batches": n_batches,
                        "completed_batches": batch_idx + 1,
                        "batch_size": batch_size,
                        "successful_extractions": successful_extractions,
                        "failed_extractions": failed_extractions,
                        "avg_time_per_item": avg_time,
                        "elapsed_time": elapsed_time
                    }
                )
            
            # Optimize memory if requested
            if optimize_after_batch:
                optimize_memory(clear_cache=True)
    
    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user")
        # Save checkpoint before exiting if checkpointing is enabled
        if checkpoint_path:
            completed_batches = len(results) // batch_size
            remaining_items = [item for batch in batches[completed_batches:] for item in batch]
            
            # Calculate final statistics
            elapsed_time = time.time() - start_time
            avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
            
            save_checkpoint(
                results=results,
                pending_items=remaining_items,
                checkpoint_path=checkpoint_path,
                metadata={
                    "model_name": model_name,
                    "field_type": field_type,
                    "total_batches": n_batches,
                    "completed_batches": completed_batches,
                    "batch_size": batch_size,
                    "interrupted": True,
                    "successful_extractions": successful_extractions,
                    "failed_extractions": failed_extractions,
                    "avg_time_per_item": avg_time,
                    "elapsed_time": elapsed_time
                }
            )
    
    finally:
        # Final checkpoint if requested
        if checkpoint_path:
            # Calculate final statistics
            elapsed_time = time.time() - start_time
            avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
            
            save_checkpoint(
                results=results,
                pending_items=[],  # Empty pending items on successful completion
                checkpoint_path=checkpoint_path,
                metadata={
                    "model_name": model_name,
                    "field_type": field_type,
                    "total_batches": n_batches,
                    "completed_batches": n_batches,
                    "batch_size": batch_size,
                    "completed": True,
                    "successful_extractions": successful_extractions,
                    "failed_extractions": failed_extractions,
                    "avg_time_per_item": avg_time,
                    "elapsed_time": elapsed_time
                }
            )
        
        # Final clean up
        optimize_memory(clear_cache=True)
        
        # Final visualization
        if show_visualization and show_progress:
            # Calculate final statistics
            elapsed_time = time.time() - start_time
            avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
            
            display_progress_visualization(
                completed=completed_items,
                total=total_items,
                successful=successful_extractions,
                failed=failed_extractions,
                avg_time_per_item=avg_time,
                elapsed_time=elapsed_time,
                batch_idx=n_batches-1,
                n_batches=n_batches,
                recent_times=processing_times[-20:] if len(processing_times) > 20 else processing_times,
                recent_errors=recent_errors
            )
        
        # Log completion
        total_time = time.time() - start_time
        items_processed = len(results)
        logger.info(f"Processing completed: {items_processed} items in {total_time:.2f}s "
                   f"({total_time/items_processed:.2f}s per item)")
        logger.info(f"Successful extractions: {successful_extractions}/{items_processed} "
                   f"({successful_extractions/items_processed*100:.2f}%)")
    
    return results


def prepare_batch_items(
    image_paths: List[Union[str, Path]],
    ground_truth_mapping: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    Prepare batch items from image paths and ground truth.
    
    Args:
        image_paths: List of paths to images
        ground_truth_mapping: Dictionary mapping image IDs to ground truth
        
    Returns:
        List of dictionaries with image_path and ground_truth
    """
    items = []
    for path in image_paths:
        # Convert to Path if it's a string
        path = Path(path) if isinstance(path, str) else path
        
        # Get image ID (stem of the filename)
        image_id = path.stem
        
        # Look up ground truth
        ground_truth = ground_truth_mapping.get(image_id, "")
        
        # Add to items list
        items.append({
            "image_id": image_id,
            "image_path": str(path),
            "ground_truth": ground_truth
        })
    
    return items