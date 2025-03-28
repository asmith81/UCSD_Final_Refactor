"""
Experiment runner for multi-field invoice data extraction.

This module provides:
- ExperimentRunner class for coordinating field extraction experiments
- Checkpointing and resumption functionality
- Result storage and organization
- Multi-field experiment coordination
"""

import os
import json
import time
import logging
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple

import torch
import numpy as np
from tqdm import tqdm

# Import project modules
from src.analysis.metrics import create_metrics_calculator
from src.execution.inference import (
    extract_field, 
    extract_work_order,
    extract_cost,
    track_gpu_memory
)
from src.results.collector import ResultsCollector


# Configure logger
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """
    Unified experiment runner for different fields.
    
    This class handles:
    - Running experiments for different fields
    - Supporting different prompt categories
    - Calculating field-specific metrics
    - Implementing checkpointing
    - Tracking experiment progress
    """
    
    def __init__(
        self, 
        experiment_config: Dict[str, Any], 
        paths: Dict[str, Any],
        model=None,
        processor=None
    ):
        """
        Initialize the experiment runner.
        
        Args:
            experiment_config: Configuration dictionary for the experiment
            paths: Dictionary of paths for loading/saving data
            model: Pre-loaded model (optional)
            processor: Pre-loaded processor (optional)
        """
        self.config = experiment_config
        self.paths = paths
        self.model = model
        self.processor = processor
        self.results = {}
        self.metrics = {}
        self.run_metadata = {
            "timestamp_start": datetime.now().isoformat(),
            "experiment_name": experiment_config.get("global", {}).get("experiment_name", "invoice_extraction"),
            "fields_processed": [],
            "prompts_processed": {},
            "total_extractions": 0,
            "errors": []
        }
        
        # Initialize metrics calculators for each field
        self.metrics_calculators = {}
        if "fields" in self.config:
            for field, field_config in self.config["fields"].items():
                try:
                    self.metrics_calculators[field] = create_metrics_calculator(field, field_config)
                    logger.info(f"Initialized metrics calculator for {field}")
                except Exception as e:
                    logger.error(f"Error initializing metrics calculator for {field}: {e}")
                    self.run_metadata["errors"].append(f"Failed to initialize metrics for {field}: {str(e)}")
        
        # Setup checkpointing
        self.checkpoint_config = self.config.get("pipeline", {}).get("checkpointing", {})
        
        # Create checkpoint directory if it doesn't exist
        checkpoint_dir = Path(self.checkpoint_config.get("checkpoint_dir", paths.get("results_dir", ".") + "/checkpoints"))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Result collector for aggregating and saving results
        self.result_collector = ResultsCollector(
            base_path=paths.get("results_dir", "."),
            experiment_name=self.run_metadata["experiment_name"]
        )
    
    def get_checkpoint_path(self, field: str) -> Path:
        """
        Get checkpoint path for a specific field.
        
        Args:
            field: Field type (work_order, cost, etc.)
            
        Returns:
            Path object for the checkpoint file
        """
        checkpoint_dir = Path(self.checkpoint_config.get(
            "checkpoint_dir", 
            str(Path(self.paths.get("results_dir", ".")) / "checkpoints")
        ))
        return checkpoint_dir / f"{field}_checkpoint.json"
    
    def load_checkpoint(self, field: str) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint for a field if available.
        
        Args:
            field: Field type to load checkpoint for
            
        Returns:
            Checkpoint data dictionary or None if not available
        """
        checkpoint_path = self.get_checkpoint_path(field)
        
        if not checkpoint_path.exists():
            return None
            
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
            logger.info(f"Loaded checkpoint for {field} from {checkpoint_path}")
            return checkpoint
        except Exception as e:
            logger.error(f"Error loading checkpoint for {field}: {e}")
            return None
    
    def save_checkpoint(self, field: str, data: Dict[str, Any]) -> None:
        """
        Save checkpoint for a field.
        
        Args:
            field: Field type to save checkpoint for
            data: Checkpoint data to save
        """
        checkpoint_path = self.get_checkpoint_path(field)
        
        try:
            with open(checkpoint_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved checkpoint for {field} to {checkpoint_path}")
        except Exception as e:
            logger.error(f"Error saving checkpoint for {field}: {e}")
    
    def run_extraction(self, batch_item: Dict[str, Any], prompt: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run extraction for a single item.
        
        Args:
            batch_item: Dictionary with image info and ground truth
            prompt: Dictionary with prompt information
            
        Returns:
            Dictionary with extraction results and metrics
        """
        field = batch_item["field_type"]
        image_path = batch_item["image_path"]
        
        # Determine model name - check all possible locations in config
        model_name = "pixtral-12b"  # Default to a known value
        
        # Try all possible config locations for model name
        if "global" in self.config and "model_name" in self.config["global"]:
            model_name = self.config["global"]["model_name"]
        elif "model_name" in self.config:
            model_name = self.config["model_name"]
        elif "experiment" in self.config and "model_name" in self.config["experiment"]:
            model_name = self.config["experiment"]["model_name"]
        
        # Log which model name we're using
        logger.info(f"Using model '{model_name}' for {field} extraction")
        
        # Track GPU memory before extraction
        pre_memory = track_gpu_memory(f"pre_extraction_{field}")
        
        # Run extraction
        try:
            # Use field-specific extraction function
            result = extract_field(
                image_path=image_path,
                field_type=field,
                prompt=prompt,
                model_name=model_name,
                model=self.model,
                processor=self.processor,
                ground_truth=batch_item.get("ground_truth", "")
            )
            
            # Track GPU memory after extraction
            post_memory = track_gpu_memory(f"post_extraction_{field}")
            
            # Add memory tracking to result
            if "metadata" not in result:
                result["metadata"] = {}
            result["metadata"]["pre_memory"] = pre_memory
            result["metadata"]["post_memory"] = post_memory
            
            # Add batch item info to result if not already present
            if "image_id" not in result:
                result["image_id"] = batch_item.get("image_id", "unknown")
            if "ground_truth" not in result:
                result["ground_truth"] = batch_item.get("ground_truth", "")
            
            # Add prompt info if not already present
            if "prompt_name" not in result:
                result["prompt_name"] = prompt.get("name", "unknown")
            if "prompt_category" not in result:
                result["prompt_category"] = prompt.get("category", "unknown")
            
            # Add processing timestamp
            result["processing_timestamp"] = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            logger.error(f"Error during extraction for {image_path}: {e}")
            error_traceback = traceback.format_exc()
            
            # Track GPU memory after error
            post_memory = track_gpu_memory(f"error_{field}")
            
            # Return error result
            return {
                "image_id": batch_item.get("image_id", "unknown"),
                "ground_truth": batch_item.get("ground_truth", ""),
                "field_type": field,
                "prompt_name": prompt.get("name", "unknown"),
                "prompt_category": prompt.get("category", "unknown"),
                "processing_timestamp": datetime.now().isoformat(),
                "error": str(e),
                "error_traceback": error_traceback,
                "metadata": {
                    "pre_memory": pre_memory,
                    "post_memory": post_memory
                }
            }
    
    def run_prompt_category(
        self, 
        field: str, 
        category: str, 
        batch_items: List[Dict[str, Any]], 
        prompts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Run extraction for all items with a specific prompt category.
        
        Args:
            field: Field type (work_order, cost, etc.)
            category: Prompt category to use
            batch_items: List of items to process
            prompts: List of available prompts
            
        Returns:
            List of extraction results
        """
        # Find matching prompt
        matching_prompt = None
        for prompt in prompts:
            if prompt["field_to_extract"] == field and prompt["category"] == category:
                matching_prompt = prompt
                break
        
        if not matching_prompt:
            error_msg = f"No prompt found for field {field}, category {category}"
            logger.error(error_msg)
            self.run_metadata["errors"].append(error_msg)
            return []
        
        # Initialize results for this category
        category_results = []
        
        # Load checkpoint if available
        checkpoint = None
        if self.checkpoint_config.get("resume_from_checkpoint", False):
            checkpoint = self.load_checkpoint(field)
        
        # Get processed image IDs from checkpoint
        processed_ids = set()
        if checkpoint and "results" in checkpoint:
            for result in checkpoint["results"]:
                if result.get("prompt_category") == category:
                    processed_ids.add(result.get("image_id"))
        
        # Filter batch items to exclude already processed ones
        remaining_items = [item for item in batch_items if item["image_id"] not in processed_ids]
        
        # If all items are processed, load results from checkpoint
        if not remaining_items:
            logger.info(f"All items already processed for {field}, {category}")
            
            # If checkpoint available, load results
            if checkpoint and "results" in checkpoint:
                return [
                    result for result in checkpoint["results"]
                    if result.get("prompt_category") == category
                ]
            return []
        
        logger.info(f"Processing {len(remaining_items)} items for {field}, {category}")
        print(f"\n🔄 Processing {len(remaining_items)} items for {field}, {category}:")
        
        # Process each item
        for i, item in enumerate(remaining_items):
            # Display progress
            print(f"  Processing item {i+1}/{len(remaining_items)}: {item['image_id']}")
            
            # Extract field
            try:
                result = self.run_extraction(item, matching_prompt)
                category_results.append(result)
                
                # Print result
                if "error" in result:
                    print(f"  ❌ Error: {result['error']}")
                else:
                    exact_match = result.get("exact_match", 0) > 0.5
                    match_symbol = "✅" if exact_match else "❌"
                    print(f"  {match_symbol} Extracted: '{result.get('processed_extraction', '')}' | "
                          f"GT: '{item['ground_truth']}' | "
                          f"Time: {result.get('processing_time', 0):.2f}s")
                
                # Update run metadata
                self.run_metadata["total_extractions"] += 1
                
                # Save checkpoint periodically
                if (i + 1) % self.checkpoint_config.get("checkpoint_frequency", 5) == 0:
                    self.save_field_results(field, category_results)
            
            except Exception as e:
                error_msg = f"Error processing {item['image_id']} for {field}, {category}: {e}"
                logger.error(error_msg)
                self.run_metadata["errors"].append(error_msg)
                
                # Create error result
                error_result = {
                    "image_id": item["image_id"],
                    "ground_truth": item["ground_truth"],
                    "field_type": field,
                    "prompt_category": category,
                    "prompt_name": matching_prompt["name"],
                    "error": str(e),
                    "processing_timestamp": datetime.now().isoformat()
                }
                
                category_results.append(error_result)
                
                # Clean up GPU memory after error
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Save final results for this category
        self.save_field_results(field, category_results)
        
        # If checkpoint available, merge with new results
        if checkpoint and "results" in checkpoint:
            # Get previously processed results for this category
            prev_results = [
                result for result in checkpoint["results"]
                if result.get("prompt_category") == category
            ]
            
            # Combine with new results
            combined_results = prev_results + category_results
            return combined_results
        
        return category_results
    
    def save_field_results(self, field: str, results: List[Dict[str, Any]]) -> None:
        """
        Save checkpoint and results for a field.
        
        Args:
            field: Field type to save results for
            results: List of results to save
        """
        # Get existing checkpoint or create new one
        checkpoint = self.load_checkpoint(field) or {
            "field": field,
            "timestamp": datetime.now().isoformat(),
            "results": []
        }
        
        # Update checkpoint with new results
        if "results" not in checkpoint:
            checkpoint["results"] = []
        
        # Add new results
        for result in results:
            # Check if result already exists
            exists = False
            for i, existing in enumerate(checkpoint["results"]):
                if (existing.get("image_id") == result.get("image_id") and
                    existing.get("prompt_category") == result.get("prompt_category")):
                    # Update existing result
                    checkpoint["results"][i] = result
                    exists = True
                    break
            
            if not exists:
                checkpoint["results"].append(result)
        
        # Update timestamp
        checkpoint["timestamp"] = datetime.now().isoformat()
        
        # Save checkpoint
        self.save_checkpoint(field, checkpoint)
        
        # Use result collector to save to results directory
        self.result_collector.save_field_results(field, checkpoint["results"])
    
    def run_field(
        self, 
        field: str, 
        batch_items: List[Dict[str, Any]], 
        prompts: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run extraction for all prompt categories for a field.
        
        Args:
            field: Field type to process
            batch_items: List of items to process
            prompts: List of available prompts
            
        Returns:
            Dictionary of results organized by prompt category
        """
        # Get field configuration
        field_config = self.config.get("fields", {}).get(field, {})
        prompt_categories = field_config.get("prompt_categories", [])
        
        # Check if prompt categories are available
        if not prompt_categories:
            logger.warning(f"No prompt categories defined for {field}")
            return {}
        
        # Initialize results for this field
        field_results = {}
        
        # Process each prompt category
        for category in prompt_categories:
            print(f"\n🚀 Running {field} extraction with {category} prompt:")
            
            # Run extraction for this category
            category_results = self.run_prompt_category(field, category, batch_items, prompts)
            
            # Store results
            field_results[category] = category_results
            
            # Update run metadata
            if field not in self.run_metadata["prompts_processed"]:
                self.run_metadata["prompts_processed"][field] = []
            
            if category not in self.run_metadata["prompts_processed"][field]:
                self.run_metadata["prompts_processed"][field].append(category)
            
            # Clean up memory between categories
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Add field to processed list
        if field not in self.run_metadata["fields_processed"]:
            self.run_metadata["fields_processed"].append(field)
        
        # Store results for this field
        self.results[field] = field_results
        
        # Calculate and store field metrics
        self.calculate_field_metrics(field)
        
        return field_results
    
    def calculate_field_metrics(self, field: str) -> Dict[str, Dict[str, float]]:
        """
        Calculate metrics for each prompt category for a field.
        
        Args:
            field: Field type to calculate metrics for
            
        Returns:
            Dictionary of metrics by prompt category
        """
        field_results = self.results.get(field, {})
        
        field_metrics = {}
        metrics_calculator = self.metrics_calculators.get(field)
        
        if not metrics_calculator:
            logger.warning(f"No metrics calculator available for {field}")
            # Create a simple metrics aggregator
            for category, results in field_results.items():
                # Skip empty results
                if not results:
                    continue
                    
                # Calculate basic metrics
                success_count = sum(1 for r in results if r.get("exact_match", 0) > 0.5)
                total_count = len(results)
                
                field_metrics[category] = {
                    "success_rate": success_count / total_count if total_count > 0 else 0,
                    "total_count": total_count,
                    "success_count": success_count,
                    "error_count": sum(1 for r in results if "error" in r)
                }
        else:
            # Use field-specific metrics calculator
            for category, results in field_results.items():
                # Skip empty results
                if not results:
                    continue
                    
                # Extract metrics from results
                metrics_list = []
                for result in results:
                    if "error" not in result:
                        # Create metrics dict from result
                        metrics_dict = {
                            metric: result.get(metric, 0)
                            for metric in ["exact_match", "character_error_rate", "levenshtein_distance", 
                                         "numeric_difference", "percentage_error"]
                            if metric in result
                        }
                        
                        # Add processing time
                        if "processing_time" in result:
                            metrics_dict["processing_time"] = result["processing_time"]
                        
                        metrics_list.append(metrics_dict)
                
                # Aggregate metrics
                category_metrics = metrics_calculator.aggregate(metrics_list)
                
                # Add success rate
                success_count = sum(1 for r in results if r.get("exact_match", 0) > 0.5)
                total_count = len(results)
                category_metrics["success_rate"] = success_count / total_count if total_count > 0 else 0
                
                # Add counts
                category_metrics["total_count"] = total_count
                category_metrics["success_count"] = success_count
                category_metrics["error_count"] = sum(1 for r in results if "error" in r)
                
                # Store metrics
                field_metrics[category] = category_metrics
        
        # Store field metrics
        self.metrics[field] = field_metrics
        
        # Save metrics using result collector
        self.result_collector.save_field_metrics(field, field_metrics)
        
        return field_metrics
    
    def run_experiment(self, fields: List[str], batch_items: Dict[str, List[Dict[str, Any]]], prompts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run complete experiment for selected fields.
        
        Args:
            fields: List of field types to process
            batch_items: Dictionary of batch items by field
            prompts: List of available prompts
            
        Returns:
            Dictionary with results and metrics
        """
        start_time = time.time()
        
        print("\n📋 Experiment Configuration:")
        print(f"  Fields: {', '.join(fields)}")
        for field in fields:
            field_config = self.config.get("fields", {}).get(field, {})
            print(f"  {field}: {', '.join(field_config.get('prompt_categories', []))} prompts")
        
        try:
            # Process each field
            for field in fields:
                print(f"\n{'=' * 40}")
                print(f"🔍 Running {field.upper()} extraction experiment")
                print(f"{'=' * 40}")
                
                # Get batch items for this field
                field_batch_items = batch_items.get(field, [])
                
                if not field_batch_items:
                    logger.warning(f"No batch items found for {field}")
                    print(f"⚠️ No batch items found for {field}, skipping...")
                    continue
                
                # Run extraction for this field
                self.run_field(field, field_batch_items, prompts)
                
                # Calculate comparative metrics across prompts for this field
                self.calculate_comparative_metrics(field)
            
            # Calculate cross-field metrics if multiple fields processed
            if len(fields) > 1:
                self.calculate_cross_field_metrics(fields)
            
            # Update run metadata
            self.run_metadata["timestamp_end"] = datetime.now().isoformat()
            self.run_metadata["total_time"] = time.time() - start_time
            
            # Save run metadata
            self.result_collector.save_run_metadata(self.run_metadata)
            
            print(f"\n✅ Experiment completed in {timedelta(seconds=int(time.time() - start_time))}")
            print(f"  Total extractions: {self.run_metadata['total_extractions']}")
            print(f"  Fields processed: {', '.join(self.run_metadata['fields_processed'])}")
            
            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Return results and metrics
            return {
                "results": self.results,
                "metrics": self.metrics,
                "metadata": self.run_metadata
            }
            
        except Exception as e:
            logger.error(f"Error running experiment: {e}")
            self.run_metadata["error"] = str(e)
            self.run_metadata["error_traceback"] = traceback.format_exc()
            
            # Save metadata even on error
            self.result_collector.save_run_metadata(self.run_metadata)
            
            return {
                "error": str(e),
                "error_traceback": traceback.format_exc(),
                "results": self.results,
                "metrics": self.metrics,
                "metadata": self.run_metadata
            }
    
    def calculate_comparative_metrics(self, field: str) -> Dict[str, Any]:
        """
        Calculate comparative metrics across prompt categories for a field.
        
        Args:
            field: Field type to calculate metrics for
            
        Returns:
            Dictionary of comparative metrics
        """
        field_metrics = self.metrics.get(field, {})
        
        if not field_metrics:
            logger.warning(f"No metrics available for field {field}")
            return {}
        
        # Identify best prompt by success rate
        best_prompt = max(
            field_metrics.items(),
            key=lambda x: x[1].get("success_rate", 0),
            default=(None, {})
        )[0]
        
        # Identify fastest prompt
        fastest_prompt = min(
            field_metrics.items(),
            key=lambda x: x[1].get("processing_time", float('inf')),
            default=(None, {})
        )[0]
        
        # Create comparison metrics
        comparison = {
            "best_prompt": best_prompt,
            "fastest_prompt": fastest_prompt,
            "metrics_by_category": field_metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save comparison metrics
        self.result_collector.save_comparative_metrics(field, comparison)
        
        return comparison
    
    def calculate_cross_field_metrics(self, fields: List[str]) -> Dict[str, Any]:
        """
        Calculate metrics comparing performance across different fields.
        
        Args:
            fields: List of fields to compare
            
        Returns:
            Dictionary of cross-field metrics
        """
        if len(fields) <= 1:
            logger.warning("Cannot calculate cross-field metrics with less than 2 fields")
            return {}
        
        # Collect metrics for each field
        cross_field_data = {}
        
        for field in fields:
            field_metrics = self.metrics.get(field, {})
            
            if not field_metrics:
                continue
                
            # Find best prompt category for this field
            best_category = max(
                field_metrics.items(),
                key=lambda x: x[1].get("success_rate", 0),
                default=(None, {})
            )[0]
            
            if best_category:
                # Get metrics for the best category
                best_metrics = field_metrics[best_category]
                
                # Store in cross-field data
                cross_field_data[field] = {
                    "best_category": best_category,
                    "success_rate": best_metrics.get("success_rate", 0),
                    "processing_time": best_metrics.get("processing_time", 0),
                    "total_count": best_metrics.get("total_count", 0),
                    "error_count": best_metrics.get("error_count", 0)
                }
        
        # Calculate overall metrics
        total_success = sum(data.get("success_rate", 0) * data.get("total_count", 0) 
                          for data in cross_field_data.values())
        total_count = sum(data.get("total_count", 0) for data in cross_field_data.values())
        
        # Create cross-field comparison
        cross_field_metrics = {
            "fields_compared": list(cross_field_data.keys()),
            "field_data": cross_field_data,
            "overall_success_rate": total_success / total_count if total_count > 0 else 0,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save cross-field metrics
        self.result_collector.save_cross_field_metrics(cross_field_metrics)
        
        return cross_field_metrics
    
    def test_extraction(self, field: str, category: str, batch_items: List[Dict[str, Any]], prompts: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Test extraction on a single image to verify functionality.
        
        Args:
            field: Field type to test
            category: Prompt category to test
            batch_items: List of batch items
            prompts: List of available prompts
            
        Returns:
            Extraction result or None if test failed
        """
        # Find matching prompt
        matching_prompt = None
        for prompt in prompts:
            if prompt["field_to_extract"] == field and prompt["category"] == category:
                matching_prompt = prompt
                break
        
        if not matching_prompt:
            logger.error(f"No prompt found for {field}, {category}")
            return None
        
        # Get a sample batch item
        if not batch_items:
            logger.error(f"No batch items found for {field}")
            return None
        
        sample_item = batch_items[0]
        
        logger.info(f"Testing {field} extraction with {category} prompt on {sample_item['image_id']}")
        print(f"🔍 Testing {field} extraction with {category} prompt:")
        print(f"  Image: {sample_item['image_id']}")
        print(f"  Ground Truth: {sample_item['ground_truth']}")
        print(f"  Prompt: \"{matching_prompt['text']}\"")
        
        # Run extraction
        result = self.run_extraction(sample_item, matching_prompt)
        
        # Print result
        if "error" in result:
            logger.error(f"Test extraction failed: {result['error']}")
            print(f"❌ Error: {result['error']}")
        else:
            logger.info(f"Test extraction succeeded: {result.get('processed_extraction', '')}")
            print(f"✅ Extraction successful:")
            print(f"  Raw extraction: \"{result.get('raw_extraction', '')}\"")
            print(f"  Processed extraction: \"{result.get('processed_extraction', '')}\"")
            print(f"  Processing time: {result.get('processing_time', 0):.2f}s")
            print(f"  Exact match: {result.get('exact_match', 0) > 0.5}")
        
        return result