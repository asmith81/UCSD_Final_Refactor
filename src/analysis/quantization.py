"""
Quantization Impact Analysis

This module provides comprehensive analysis of quantization strategies for vision-language models,
including performance benchmarking, memory usage tracking, speed-accuracy trade-offs,
and hardware-adaptive optimization.

The analysis focuses on understanding the impact of different quantization approaches
on extraction accuracy, inference speed, and memory efficiency to enable optimal
configuration for various deployment environments.
"""

import os
import time
import logging
import gc
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable

import numpy as np
import torch
import matplotlib.pyplot as plt

# Import project components
from src.config.experiment import ExperimentConfiguration
from src.models.model_service import get_model_service, ModelLoadingResult, optimize_memory
from src.data.loader import load_and_prepare_data
from src.prompts.registry import get_prompt_registry
from src.execution.inference import process_image_with_metrics
from src.analysis.metrics import calculate_batch_metrics
from src.results.storage import ResultStorage

# Configure logging
logger = logging.getLogger(__name__)


class QuantizationPriority(Enum):
    """Priorities for selecting quantization strategies."""
    SPEED = auto()         # Prioritize inference speed
    MEMORY = auto()        # Prioritize memory efficiency
    ACCURACY = auto()      # Prioritize extraction accuracy
    BALANCED = auto()      # Balanced approach considering all factors
    
    @classmethod
    def from_string(cls, value: str) -> 'QuantizationPriority':
        """Convert string to enum value."""
        try:
            return cls[value.upper()]
        except (KeyError, TypeError):
            logger.warning(f"Unknown quantization priority: {value}. Using BALANCED.")
            return cls.BALANCED


@dataclass
class QuantizationScoreWeights:
    """Weights for calculating efficiency scores based on priorities."""
    memory_weight: float = 0.33
    speed_weight: float = 0.33
    accuracy_weight: float = 0.34
    
    @classmethod
    def for_priority(cls, priority: QuantizationPriority) -> 'QuantizationScoreWeights':
        """Create weight configuration based on priority."""
        if priority == QuantizationPriority.MEMORY:
            return cls(memory_weight=0.6, speed_weight=0.2, accuracy_weight=0.2)
        elif priority == QuantizationPriority.SPEED:
            return cls(memory_weight=0.2, speed_weight=0.6, accuracy_weight=0.2)
        elif priority == QuantizationPriority.ACCURACY:
            return cls(memory_weight=0.2, speed_weight=0.2, accuracy_weight=0.6)
        else:  # BALANCED or unknown
            return cls()  # Default weights


@dataclass
class HardwareProfile:
    """Hardware capabilities profile for strategy selection."""
    has_gpu: bool = False
    gpu_memory_gb: float = 0.0
    cpu_memory_gb: float = 0.0
    cuda_compute_capability: Optional[str] = None
    gpu_name: str = "CPU only"
    
    @classmethod
    def detect_current(cls) -> 'HardwareProfile':
        """Detect current hardware capabilities."""
        profile = cls()
        
        # Check for GPU
        if torch.cuda.is_available():
            profile.has_gpu = True
            device = torch.cuda.current_device()
            
            # Get GPU properties
            props = torch.cuda.get_device_properties(device)
            profile.gpu_memory_gb = props.total_memory / (1024 ** 3)
            profile.cuda_compute_capability = f"{props.major}.{props.minor}"
            profile.gpu_name = torch.cuda.get_device_name(device)
        
        # Try to get system memory
        try:
            import psutil
            mem = psutil.virtual_memory()
            profile.cpu_memory_gb = mem.total / (1024 ** 3)
        except ImportError:
            # Fallback to reasonable default
            profile.cpu_memory_gb = 16.0
            logger.warning("psutil not available, using default CPU memory estimate")
        
        return profile
    
    def supports_strategy(self, strategy: str, model_memory_requirements: Dict[str, float]) -> bool:
        """Check if hardware supports a quantization strategy."""
        # CPU-only system can only use CPU-compatible strategies
        if not self.has_gpu and strategy not in ['none', 'cpu', 'int8-cpu']:
            return False
        
        # Check memory requirements
        if strategy in model_memory_requirements:
            required_memory = model_memory_requirements[strategy]
            
            # For GPU strategies
            if strategy not in ['none', 'cpu', 'int8-cpu']:
                # Need some headroom - 90% of available memory
                available_memory = self.gpu_memory_gb * 0.9
                return required_memory <= available_memory
            else:
                # CPU strategies - need more headroom - 80% of available memory
                available_memory = self.cpu_memory_gb * 0.8
                return required_memory <= available_memory
                
        return True  # Assume supported if no specific requirements
    
    def get_supported_strategies(
        self,
        strategies: List[str],
        model_memory_requirements: Dict[str, float]
    ) -> List[str]:
        """Filter strategies supported by current hardware."""
        return [
            strategy for strategy in strategies 
            if self.supports_strategy(strategy, model_memory_requirements)
        ]


@dataclass
class QuantizationResult:
    """Results from benchmarking a single quantization strategy."""
    strategy: str
    bits: int
    loading_time: float
    memory_used_gb: float
    inference_time: float
    accuracy: float
    efficiency_score: float = 0.0
    hardware_compatibility: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "strategy": self.strategy,
            "bits": self.bits,
            "loading_time": self.loading_time,
            "memory_used_gb": self.memory_used_gb,
            "inference_time": self.inference_time,
            "accuracy": self.accuracy,
            "efficiency_score": self.efficiency_score,
            "hardware_compatibility": self.hardware_compatibility,
            "error": self.error
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QuantizationResult':
        """Create from dictionary."""
        return cls(
            strategy=data.get("strategy", "unknown"),
            bits=data.get("bits", 0),
            loading_time=data.get("loading_time", 0.0),
            memory_used_gb=data.get("memory_used_gb", 0.0),
            inference_time=data.get("inference_time", 0.0),
            accuracy=data.get("accuracy", 0.0),
            efficiency_score=data.get("efficiency_score", 0.0),
            hardware_compatibility=data.get("hardware_compatibility", {}),
            error=data.get("error")
        )


@dataclass
class QuantizationBenchmark:
    """Complete benchmark results across multiple quantization strategies."""
    model_name: str
    timestamp: datetime = field(default_factory=datetime.now)
    hardware_info: Dict[str, Any] = field(default_factory=dict)
    results: List[QuantizationResult] = field(default_factory=list)
    best_strategy: Optional[str] = None
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "timestamp": self.timestamp.isoformat(),
            "hardware_info": self.hardware_info,
            "results": [r.to_dict() for r in self.results],
            "best_strategy": self.best_strategy,
            "recommendations": self.recommendations
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QuantizationBenchmark':
        """Create from dictionary."""
        # Parse timestamp
        timestamp = datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat()))
        
        # Create instance
        benchmark = cls(
            model_name=data.get("model_name", "unknown"),
            timestamp=timestamp,
            hardware_info=data.get("hardware_info", {}),
            best_strategy=data.get("best_strategy"),
            recommendations=data.get("recommendations", [])
        )
        
        # Parse results
        results_data = data.get("results", [])
        benchmark.results = [QuantizationResult.from_dict(r) for r in results_data]
        
        return benchmark


class QuantizationAnalyzer:
    """
    Comprehensive analyzer for quantization strategy impact.
    
    This class provides methods for:
    - Benchmarking different quantization strategies
    - Analyzing memory usage patterns
    - Computing speed-accuracy trade-offs
    - Selecting optimal strategies based on hardware
    - Generating recommendations for deployment
    """
    
    def __init__(
        self,
        model_name: str,
        config: Optional[ExperimentConfiguration] = None,
        results_dir: Optional[Union[str, Path]] = None,
        sample_size: int = 10,
        timeout: int = 600  # 10 minutes max for full benchmark
    ):
        """
        Initialize quantization analyzer.
        
        Args:
            model_name: Name of the model to analyze
            config: Optional experiment configuration
            results_dir: Directory to store results
            sample_size: Number of images to use for benchmarking
            timeout: Maximum time (seconds) for complete benchmark
        """
        self.model_name = model_name
        self.config = config
        self.sample_size = min(sample_size, 50)  # Cap at 50 for reasonable time
        self.timeout = timeout
        
        # Set up model service
        self.model_service = get_model_service()
        
        # Set up results storage
        if results_dir:
            self.results_dir = Path(results_dir)
        elif config:
            self.results_dir = Path(config.get('results_dir', 'results'))
        else:
            self.results_dir = Path('results')
        
        self.results_dir = self.results_dir / 'quantization'
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Set up storage
        self.storage = ResultStorage(
            base_path=self.results_dir,
            experiment_name=f"quant_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Initialize sample data
        self.sample_data = None
        self.prompts = None
        
        # Hardware profile for compatibility checks
        self.hardware = HardwareProfile.detect_current()
        logger.info(f"Detected hardware: {self.hardware.gpu_name}, "
                    f"Memory: {self.hardware.gpu_memory_gb:.2f}GB")
    
    def prepare_benchmark_data(self) -> bool:
        """
        Prepare a small dataset for benchmarking.
        
        Returns:
            True if data preparation successful, False otherwise
        """
        try:
            # Make sure we have configuration
            if not self.config:
                logger.warning("No configuration provided, using defaults")
                # Create minimal default configuration
                from src.config.base_config import ConfigurationManager
                config_manager = ConfigurationManager()
                self.config = ExperimentConfiguration(
                    name="quantization_benchmark",
                    fields_to_extract=["work_order"],
                    model_name=self.model_name
                )
            
            # Get data paths from configuration
            ground_truth_path = self.config.get('ground_truth_path')
            if not ground_truth_path or not os.path.exists(ground_truth_path):
                logger.error("Ground truth file not found")
                return False
                
            image_dir = self.config.get('images_dir')
            if not image_dir or not os.path.exists(image_dir):
                logger.error("Image directory not found")
                return False
            
            # Determine which field to use for benchmarking
            field = self.config.get('fields_to_extract', ['work_order'])[0]
            
            # Get field-specific column name from configuration
            field_column = None
            extraction_fields = self.config.get('extraction_fields', {})
            if field in extraction_fields and 'csv_column_name' in extraction_fields[field]:
                field_column = extraction_fields[field]['csv_column_name']
                
            # Load a small dataset for benchmarking
            _, ground_truth_mapping, batch_items = load_and_prepare_data(
                ground_truth_path=ground_truth_path,
                image_dir=image_dir,
                field_to_extract=field,
                field_column_name=field_column or f"{field} Number",
                image_id_column='Invoice'
            )
            
            # Limit to sample size
            if batch_items:
                self.sample_data = batch_items[:self.sample_size]
                logger.info(f"Prepared {len(self.sample_data)} sample items for benchmarking")
            else:
                logger.error("No batch items found for benchmarking")
                return False
            
            # Get prompts for benchmarking
            prompt_registry = get_prompt_registry()
            self.prompts = prompt_registry.get_by_field(field)
            
            if not self.prompts:
                logger.warning(f"No prompts found for field {field}, creating default")
                # Create a basic default prompt
                from src.prompts.registry import Prompt
                default_prompt = Prompt(
                    name=f'default_{field}',
                    text=f'Extract the {field.replace("_", " ")} from this invoice image.',
                    field_to_extract=field,
                    category='basic'
                )
                self.prompts = [default_prompt]
            
            # Use just the first prompt for benchmarking
            self.prompts = [self.prompts[0]]
            logger.info(f"Using prompt: {self.prompts[0].name}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error preparing benchmark data: {e}")
            return False
    
    def run_benchmark(
        self,
        strategies: Optional[List[str]] = None,
        priority: Union[str, QuantizationPriority] = QuantizationPriority.BALANCED
    ) -> Optional[QuantizationBenchmark]:
        """
        Run comprehensive benchmark across quantization strategies.
        
        Args:
            strategies: List of quantization strategies to benchmark
                       (None for all available strategies)
            priority: Priority for efficiency scoring
            
        Returns:
            QuantizationBenchmark results or None if benchmark failed
        """
        # Prepare data if not already done
        if not self.sample_data:
            success = self.prepare_benchmark_data()
            if not success:
                logger.error("Failed to prepare benchmark data")
                return None
        
        # Get model configuration to determine available strategies
        model_config = self.model_service.get_model_config(self.model_name)
        if not model_config:
            logger.error(f"Model configuration not found for {self.model_name}")
            return None
            
        # Get available strategies
        available_strategies = model_config.get_available_quantization_strategies()
        logger.info(f"Available strategies: {', '.join(available_strategies)}")
        
        # Use specified strategies or all available
        if strategies:
            benchmark_strategies = [s for s in strategies if s in available_strategies]
            if len(benchmark_strategies) < len(strategies):
                logger.warning(f"Some specified strategies are not available for {self.model_name}")
        else:
            benchmark_strategies = available_strategies
            
        # Filter strategies based on hardware compatibility
        memory_requirements = model_config.get_strategy_memory_requirements()
        compatible_strategies = self.hardware.get_supported_strategies(
            benchmark_strategies, memory_requirements
        )
        
        if len(compatible_strategies) < len(benchmark_strategies):
            logger.warning(f"{len(benchmark_strategies) - len(compatible_strategies)} "
                          f"strategies incompatible with current hardware")
        
        if not compatible_strategies:
            logger.error("No compatible quantization strategies found for current hardware")
            return None
            
        # Convert priority string to enum if needed
        if isinstance(priority, str):
            priority = QuantizationPriority.from_string(priority)
            
        # Create benchmark result
        benchmark = QuantizationBenchmark(
            model_name=self.model_name,
            hardware_info={
                "gpu_name": self.hardware.gpu_name,
                "gpu_memory_gb": self.hardware.gpu_memory_gb,
                "cpu_memory_gb": self.hardware.cpu_memory_gb,
                "cuda_compute_capability": self.hardware.cuda_compute_capability,
                "has_gpu": self.hardware.has_gpu
            }
        )
        
        # Set up weights for efficiency scoring
        weights = QuantizationScoreWeights.for_priority(priority)
        logger.info(f"Using weights - Memory: {weights.memory_weight}, "
                   f"Speed: {weights.speed_weight}, Accuracy: {weights.accuracy_weight}")
        
        # Track start time for timeout
        start_time = time.time()
        
        # Benchmark each strategy
        for strategy in compatible_strategies:
            # Check timeout
            if time.time() - start_time > self.timeout:
                logger.warning(f"Benchmark timeout reached after {self.timeout}s, "
                              f"skipping remaining strategies")
                break
                
            logger.info(f"Benchmarking strategy: {strategy}")
            
            try:
                # Benchmark this strategy
                result = self._benchmark_strategy(strategy, model_config)
                if result:
                    # Calculate efficiency score
                    result.efficiency_score = self._calculate_efficiency_score(
                        result, weights, benchmark.results
                    )
                    benchmark.results.append(result)
                    logger.info(f"Strategy {strategy} - Efficiency score: {result.efficiency_score:.4f}")
            except Exception as e:
                logger.error(f"Error benchmarking strategy {strategy}: {e}")
                # Add error result
                error_result = QuantizationResult(
                    strategy=strategy,
                    bits=model_config.get_bits_for_strategy(strategy),
                    loading_time=0.0,
                    memory_used_gb=0.0,
                    inference_time=0.0,
                    accuracy=0.0,
                    error=str(e)
                )
                benchmark.results.append(error_result)
        
        # Find best strategy based on efficiency score
        if benchmark.results:
            # Sort by efficiency score (descending)
            sorted_results = sorted(
                benchmark.results,
                key=lambda r: r.efficiency_score if r.error is None else -1.0,
                reverse=True
            )
            
            # Get best strategy
            best_result = sorted_results[0]
            if best_result.error is None:
                benchmark.best_strategy = best_result.strategy
                logger.info(f"Best strategy: {benchmark.best_strategy} "
                          f"(score: {best_result.efficiency_score:.4f})")
            
            # Generate recommendations
            self._generate_recommendations(benchmark, priority)
            
            # Save benchmark results
            self._save_benchmark_results(benchmark)
            
            return benchmark
        else:
            logger.error("No benchmark results generated")
            return None
    
    def _benchmark_strategy(
        self,
        strategy: str,
        model_config: Any
    ) -> Optional[QuantizationResult]:
        """
        Benchmark a single quantization strategy.
        
        Args:
            strategy: Quantization strategy to benchmark
            model_config: Model configuration
            
        Returns:
            QuantizationResult or None if benchmark failed
        """
        # Clean up before benchmark
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Record memory before loading
        memory_before = self._get_gpu_memory()
        
        # Load model with strategy
        load_start = time.time()
        loading_result = self.model_service.load_model(
            model_name=self.model_name,
            quantization=strategy,
            force_reload=True
        )
        loading_time = time.time() - load_start
        
        # Handle loading errors
        if not loading_result.success:
            logger.error(f"Failed to load model with strategy {strategy}: {loading_result.error}")
            return QuantizationResult(
                strategy=strategy,
                bits=model_config.get_bits_for_strategy(strategy),
                loading_time=loading_time,
                memory_used_gb=0.0,
                inference_time=0.0,
                accuracy=0.0,
                error=str(loading_result.error)
            )
        
        # Record memory after loading
        memory_after = self._get_gpu_memory()
        memory_used = memory_after - memory_before
        
        # Run inference benchmark
        inference_times = []
        extraction_results = []
        
        # Use only the first prompt for benchmarking
        prompt = self.prompts[0]
        
        for idx, item in enumerate(self.sample_data):
            try:
                # Measure inference time
                inference_start = time.time()
                result = process_image_with_metrics(
                    image_path=item['image_path'],
                    ground_truth=item['ground_truth'],
                    prompt=prompt.to_dict(),
                    model_name=self.model_name,
                    field_type=prompt.field_to_extract,
                    model=loading_result.model,
                    processor=loading_result.processor
                )
                inference_time = time.time() - inference_start
                
                # Track results
                inference_times.append(inference_time)
                extraction_results.append(result)
                
                # Log progress periodically
                if (idx + 1) % 5 == 0 or idx == 0 or idx == len(self.sample_data) - 1:
                    logger.info(f"Processed item {idx+1}/{len(self.sample_data)} "
                               f"for strategy: {strategy}")
                
            except Exception as e:
                logger.error(f"Error processing item {idx}: {e}")
        
        # Calculate metrics
        try:
            metrics = calculate_batch_metrics(
                extraction_results, 
                field=prompt.field_to_extract,
                config={}
            )
            
            # Calculate average inference time
            avg_inference_time = np.mean(inference_times) if inference_times else float('inf')
            
            # Create result
            result = QuantizationResult(
                strategy=strategy,
                bits=model_config.get_bits_for_strategy(strategy),
                loading_time=loading_time,
                memory_used_gb=memory_used,
                inference_time=avg_inference_time,
                accuracy=metrics.get('success_rate', 0.0),
                hardware_compatibility={
                    "compatible": True,
                    "exceeds_memory": False
                }
            )
            
            # Check if memory usage exceeds hardware limits
            if self.hardware.has_gpu and memory_used > (self.hardware.gpu_memory_gb * 0.9):
                result.hardware_compatibility["exceeds_memory"] = True
                result.hardware_compatibility["compatible"] = False
                
            # Log result
            logger.info(f"Strategy {strategy}: "
                       f"Memory={memory_used:.2f}GB, "
                       f"Loading={loading_time:.2f}s, "
                       f"Inference={avg_inference_time:.4f}s, "
                       f"Accuracy={metrics.get('success_rate', 0.0):.2%}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating metrics for strategy {strategy}: {e}")
            return None
        finally:
            # Clean up
            self.model_service.unload_model(self.model_name)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    def _calculate_efficiency_score(
        self,
        result: QuantizationResult,
        weights: QuantizationScoreWeights,
        previous_results: List[QuantizationResult]
    ) -> float:
        """
        Calculate weighted efficiency score for a strategy.
        
        Args:
            result: Benchmark result to score
            weights: Scoring weights
            previous_results: Previously benchmarked strategies for normalization
            
        Returns:
            Efficiency score (0.0 to 1.0, higher is better)
        """
        # Get min/max values across all results including current one
        all_results = previous_results + [result]
        
        # Skip results with errors for normalization
        valid_results = [r for r in all_results if r.error is None]
        if not valid_results:
            return 0.0
            
        # Get min/max values for normalization
        memory_values = [r.memory_used_gb for r in valid_results]
        min_memory = min(memory_values) if memory_values else 0.0
        max_memory = max(memory_values) if memory_values else 1.0
        
        inference_values = [r.inference_time for r in valid_results]
        min_inference = min(inference_values) if inference_values else 0.0
        max_inference = max(inference_values) if inference_values else 1.0
        
        # Prevent division by zero
        memory_range = max_memory - min_memory
        if memory_range <= 0:
            memory_range = 1.0
            
        inference_range = max_inference - min_inference
        if inference_range <= 0:
            inference_range = 1.0
        
        # Calculate normalized scores (0 to 1, higher is better)
        memory_score = 1.0 - ((result.memory_used_gb - min_memory) / memory_range)
        speed_score = 1.0 - ((result.inference_time - min_inference) / inference_range)
        accuracy_score = result.accuracy  # Already 0 to 1
        
        # Calculate weighted score
        efficiency_score = (
            weights.memory_weight * memory_score +
            weights.speed_weight * speed_score +
            weights.accuracy_weight * accuracy_score
        )
        
        return efficiency_score
    
    def _generate_recommendations(
        self,
        benchmark: QuantizationBenchmark,
        priority: QuantizationPriority
    ) -> None:
        """
        Generate recommendations based on benchmark results.
        
        Args:
            benchmark: Benchmark results
            priority: Priority used for scoring
        """
        if not benchmark.results:
            benchmark.recommendations.append("No valid quantization strategies benchmarked")
            return
            
        # Filter valid results
        valid_results = [r for r in benchmark.results if r.error is None]
        if not valid_results:
            benchmark.recommendations.append("All benchmarked strategies failed")
            return
            
        # Sort by efficiency score
        sorted_results = sorted(valid_results, key=lambda r: r.efficiency_score, reverse=True)
        
        # Add general recommendation
        benchmark.recommendations.append(
            f"Recommended strategy: {benchmark.best_strategy} with "
            f"efficiency score {sorted_results[0].efficiency_score:.4f}"
        )
        
        # Add priority-specific recommendation
        if priority == QuantizationPriority.MEMORY:
            # Find most memory-efficient strategy with acceptable accuracy
            memory_efficient = sorted(valid_results, key=lambda r: r.memory_used_gb)
            for result in memory_efficient:
                if result.accuracy >= 0.7:  # Accept 70%+ accuracy
                    benchmark.recommendations.append(
                        f"Most memory-efficient strategy with good accuracy: {result.strategy} "
                        f"using {result.memory_used_gb:.2f}GB with {result.accuracy:.1%} accuracy"
                    )
                    break
        
        elif priority == QuantizationPriority.SPEED:
            # Find fastest strategy with acceptable accuracy
            speed_efficient = sorted(valid_results, key=lambda r: r.inference_time)
            for result in speed_efficient:
                if result.accuracy >= 0.7:  # Accept 70%+ accuracy
                    benchmark.recommendations.append(
                        f"Fastest strategy with good accuracy: {result.strategy} "
                        f"at {result.inference_time:.4f}s/image with {result.accuracy:.1%} accuracy"
                    )
                    break
        
        elif priority == QuantizationPriority.ACCURACY:
            # Find most accurate strategy
            accuracy_sorted = sorted(valid_results, key=lambda r: r.accuracy, reverse=True)
            benchmark.recommendations.append(
                f"Most accurate strategy: {accuracy_sorted[0].strategy} "
                f"with {accuracy_sorted[0].accuracy:.1%} accuracy"
            )
        
        # Add hardware-specific recommendations
        if not self.hardware.has_gpu:
            benchmark.recommendations.append(
                "CPU-only system detected. For better performance, consider using a GPU."
            )
        elif sorted_results[0].memory_used_gb > (self.hardware.gpu_memory_gb * 0.8):
            benchmark.recommendations.append(
                f"Even the best strategy ({benchmark.best_strategy}) "
                f"uses {sorted_results[0].memory_used_gb:.2f}GB, which is close to your "
                f"GPU's {self.hardware.gpu_memory_gb:.2f}GB capacity. Consider "
                f"lower-precision strategies or smaller batch sizes."
            )
    
    def _save_benchmark_results(self, benchmark: QuantizationBenchmark) -> None:
        """
        Save benchmark results to file.
        
        Args:
            benchmark: Benchmark results to save
        """
        # Save to JSON file directly
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.model_name}_quantization_benchmark_{timestamp}.json"
        file_path = self.results_dir / filename
        
        with open(file_path, 'w') as f:
            import json
            json.dump(benchmark.to_dict(), f, indent=2)
        
        logger.info(f"Saved benchmark results to {file_path}")
        
        # Also save to result storage if available
        try:
            self.storage.save_results(benchmark.to_dict(), format='json')
            logger.info(f"Also saved benchmark results to experiment storage")
        except Exception as e:
            logger.warning(f"Could not save to experiment storage: {e}")
    
    def _get_gpu_memory(self) -> float:
        """
        Get current GPU memory usage in GB.
        
        Returns:
            Memory used in GB
        """
        if not torch.cuda.is_available():
            return 0.0
            
        try:
            # Get current memory usage - allocated + cached memory
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            
            # For more detailed analysis, you can get per-device stats
            device_count = torch.cuda.device_count()
            device_stats = {}
            
            for i in range(device_count):
                device_allocated = torch.cuda.memory_allocated(i)
                device_reserved = torch.cuda.memory_reserved(i)
                device_stats[f"cuda:{i}"] = {
                    "allocated_gb": device_allocated / (1024 ** 3),
                    "reserved_gb": device_reserved / (1024 ** 3),
                    "total_gb": (device_allocated + device_reserved) / (1024 ** 3)
                }
            
            # Store device stats for later analysis
            self._last_device_memory_stats = device_stats
            
            # Return total memory used across all devices in GB
            return allocated / (1024 ** 3)
        except Exception as e:
            logger.warning(f"Error getting GPU memory: {e}")
            return 0.0
    
    def _track_memory_usage(self, strategy: str) -> Dict[str, float]:
        """
        Track memory usage during model loading and inference.
        
        Args:
            strategy: Quantization strategy being tested
            
        Returns:
            Dictionary with memory usage metrics
        """
        memory_metrics = {
            "before_loading": 0.0,
            "after_loading": 0.0,
            "peak_during_inference": 0.0,
            "after_inference": 0.0,
            "after_cleanup": 0.0,
            "model_only": 0.0,
            "inference_overhead": 0.0,
            "memory_timeline": [],
            "detailed_stats": {}
        }
        
        try:
            # Record initial memory
            memory_metrics["before_loading"] = self._get_gpu_memory()
            start_time = time.time()
            
            # Record memory during model loading
            memory_timeline = []
            
            # Load model and record memory
            loading_result = self.model_service.load_model(
                model_name=self.model_name,
                quantization=strategy,
                force_reload=True
            )
            
            memory_metrics["after_loading"] = self._get_gpu_memory()
            memory_metrics["model_only"] = memory_metrics["after_loading"] - memory_metrics["before_loading"]
            
            # Add loading details to timeline
            memory_timeline.append({
                "timestamp": time.time() - start_time,
                "phase": "model_loaded",
                "memory_gb": memory_metrics["after_loading"]
            })
            
            # Store model loading details if available
            if hasattr(loading_result, "metrics"):
                memory_metrics["detailed_stats"]["loading"] = loading_result.metrics
            
            # Track peak memory during inference
            peak_memory = memory_metrics["after_loading"]
            inference_times = []
            
            for idx, item in enumerate(self.sample_data):
                try:
                    inference_start = time.time()
                    
                    # Run inference
                    result = process_image_with_metrics(
                        image_path=item['image_path'],
                        ground_truth=item['ground_truth'],
                        prompt=self.prompts[0].to_dict(),
                        model_name=self.model_name,
                        field_type=self.prompts[0].field_to_extract
                    )
                    
                    inference_end = time.time()
                    inference_times.append(inference_end - inference_start)
                    
                    # Update peak memory
                    current_memory = self._get_gpu_memory()
                    peak_memory = max(peak_memory, current_memory)
                    
                    # Add to timeline
                    memory_timeline.append({
                        "timestamp": inference_end - start_time,
                        "phase": f"inference_{idx}",
                        "memory_gb": current_memory,
                        "inference_time": inference_end - inference_start
                    })
                    
                    # Optional - if there's device-specific data available
                    if hasattr(self, "_last_device_memory_stats"):
                        memory_timeline[-1]["device_stats"] = self._last_device_memory_stats
                    
                except Exception as e:
                    logger.warning(f"Error during inference: {e}")
            
            memory_metrics["peak_during_inference"] = peak_memory
            memory_metrics["after_inference"] = self._get_gpu_memory()
            memory_metrics["inference_overhead"] = memory_metrics["peak_during_inference"] - memory_metrics["after_loading"]
            
            # Add inference stats
            if inference_times:
                memory_metrics["detailed_stats"]["inference"] = {
                    "mean_time": sum(inference_times) / len(inference_times),
                    "min_time": min(inference_times),
                    "max_time": max(inference_times),
                    "samples": len(inference_times)
                }
            
            # Cleanup and record final memory
            memory_timeline.append({
                "timestamp": time.time() - start_time,
                "phase": "before_cleanup",
                "memory_gb": self._get_gpu_memory()
            })
            
            self.model_service.unload_model(self.model_name)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            memory_metrics["after_cleanup"] = self._get_gpu_memory()
            
            # Record final timeline point
            memory_timeline.append({
                "timestamp": time.time() - start_time,
                "phase": "after_cleanup",
                "memory_gb": memory_metrics["after_cleanup"]
            })
            
            memory_metrics["memory_timeline"] = memory_timeline
            
        except Exception as e:
            logger.error(f"Error tracking memory usage: {e}")
        
        return memory_metrics
    
    def _generate_memory_analysis(self, benchmark: QuantizationBenchmark) -> Dict[str, Any]:
        """
        Generate detailed memory analysis from benchmark results.
        
        Args:
            benchmark: Benchmark results to analyze
            
        Returns:
            Dictionary with memory analysis details
        """
        analysis = {
            "total_memory_usage": {},
            "memory_savings": {},
            "fragmentation_analysis": {},
            "memory_efficiency": {},
            "timeline_analysis": {},
            "overhead_analysis": {},
            "recommendations": []
        }
        
        # Filter valid results
        valid_results = [r for r in benchmark.results if r.error is None]
        if not valid_results:
            return analysis
        
        # Find baseline (no quantization) for comparison
        baseline_strategy = next((r.strategy for r in valid_results if r.strategy == "none"), None)
        if not baseline_strategy:
            # If none exists, use the first result as baseline
            baseline_strategy = valid_results[0].strategy
        
        baseline_result = next((r for r in valid_results if r.strategy == baseline_strategy), None)
        
        # Calculate total memory usage per strategy
        for result in valid_results:
            # If we have detailed stats, use them
            if hasattr(result, "detailed_stats") and result.detailed_stats:
                memory_data = result.detailed_stats.get("memory", {})
                analysis["total_memory_usage"][result.strategy] = {
                    "loading": memory_data.get("model_only", result.memory_used_gb),
                    "inference": memory_data.get("peak_during_inference", result.memory_used_gb),
                    "overhead": memory_data.get("inference_overhead", 0.0),
                    "total": result.memory_used_gb
                }
            else:
                analysis["total_memory_usage"][result.strategy] = {
                    "loading": result.memory_used_gb,
                    "inference": result.memory_used_gb,
                    "total": result.memory_used_gb
                }
        
        # Calculate memory savings relative to baseline
        if baseline_result and baseline_strategy in analysis["total_memory_usage"]:
            baseline_memory = analysis["total_memory_usage"][baseline_strategy]["total"]
            for strategy, usage in analysis["total_memory_usage"].items():
                if strategy != baseline_strategy:
                    savings = baseline_memory - usage["total"]
                    savings_percent = (savings / baseline_memory) * 100 if baseline_memory > 0 else 0
                    analysis["memory_savings"][strategy] = {
                        "absolute_gb": savings,
                        "percentage": savings_percent,
                        "relative_size": f"{usage['total'] / baseline_memory:.2f}x smaller" 
                            if baseline_memory > 0 and usage['total'] > 0 else "N/A"
                    }
        
        # Analyze memory fragmentation
        for result in valid_results:
            fragmentation_data = {}
            
            if hasattr(result, "hardware_compatibility"):
                fragmentation_data.update({
                    "compatible": result.hardware_compatibility.get("compatible", True),
                    "exceeds_memory": result.hardware_compatibility.get("exceeds_memory", False)
                })
            
            # Check if we have detailed memory stats
            if hasattr(result, "detailed_stats") and result.detailed_stats:
                memory_stats = result.detailed_stats.get("memory", {})
                
                # Calculate fragmentation as (reserved - allocated) / reserved
                if "after_loading" in memory_stats and "after_cleanup" in memory_stats:
                    retained = max(0, memory_stats["after_cleanup"] - memory_stats["before_loading"])
                    fragmentation_data["retained_memory_gb"] = retained
                    
                    if memory_stats["after_loading"] > 0:
                        fragmentation_ratio = retained / memory_stats["after_loading"]
                        fragmentation_data["fragmentation_ratio"] = fragmentation_ratio
                        fragmentation_data["fragmentation_percentage"] = fragmentation_ratio * 100
            
            analysis["fragmentation_analysis"][result.strategy] = fragmentation_data
        
        # Calculate memory efficiency (accuracy per GB)
        for result in valid_results:
            if result.memory_used_gb > 0:
                efficiency = result.accuracy / result.memory_used_gb
                analysis["memory_efficiency"][result.strategy] = {
                    "accuracy_per_gb": efficiency,
                    "relative_efficiency": 1.0  # Will update after computing all
                }
        
        # Calculate relative efficiency compared to best
        if analysis["memory_efficiency"]:
            best_efficiency = max(
                data["accuracy_per_gb"] 
                for data in analysis["memory_efficiency"].values()
            )
            
            for strategy in analysis["memory_efficiency"]:
                current = analysis["memory_efficiency"][strategy]["accuracy_per_gb"]
                if best_efficiency > 0:
                    analysis["memory_efficiency"][strategy]["relative_efficiency"] = current / best_efficiency
        
        # Timeline analysis if available
        for result in valid_results:
            if hasattr(result, "detailed_stats") and "memory_timeline" in result.detailed_stats:
                timeline = result.detailed_stats["memory_timeline"]
                
                # Extract key phases
                key_points = {}
                for point in timeline:
                    phase = point.get("phase", "")
                    if phase in ["model_loaded", "before_cleanup", "after_cleanup"]:
                        key_points[phase] = point
                
                analysis["timeline_analysis"][result.strategy] = {
                    "timestamps": [p.get("timestamp", 0) for p in timeline],
                    "memory_values": [p.get("memory_gb", 0) for p in timeline],
                    "key_points": key_points
                }
        
        # Overhead analysis - analyze memory overhead during inference
        for result in valid_results:
            if hasattr(result, "detailed_stats") and "memory" in result.detailed_stats:
                memory_stats = result.detailed_stats["memory"]
                
                # Calculate memory growth during inference
                if "after_loading" in memory_stats and "peak_during_inference" in memory_stats:
                    base = memory_stats["after_loading"]
                    peak = memory_stats["peak_during_inference"]
                    
                    if base > 0:
                        growth_ratio = peak / base
                        growth_pct = (peak - base) / base * 100
                        
                        analysis["overhead_analysis"][result.strategy] = {
                            "base_memory_gb": base,
                            "peak_memory_gb": peak,
                            "growth_gb": peak - base,
                            "growth_ratio": growth_ratio,
                            "growth_percentage": growth_pct
                        }
        
        # Generate memory-specific recommendations
        if self.hardware.has_gpu:
            available_memory = self.hardware.gpu_memory_gb
            
            # Check if any strategy is close to memory limits
            for strategy, usage in analysis["total_memory_usage"].items():
                if usage["total"] > (available_memory * 0.8):
                    analysis["recommendations"].append(
                        f"Strategy {strategy} uses {usage['total']:.2f}GB, which is close to "
                        f"the available {available_memory:.2f}GB. Consider using a more aggressive "
                        "quantization approach."
                    )
            
            # Find strategies with best memory efficiency
            if analysis["memory_efficiency"]:
                best_efficiency_strategy = max(
                    analysis["memory_efficiency"].items(),
                    key=lambda x: x[1]["accuracy_per_gb"]
                )[0]
                
                analysis["recommendations"].append(
                    f"Strategy {best_efficiency_strategy} provides the best accuracy per GB "
                    "of memory used. Consider this option for memory-constrained environments."
                )
            
            # Check for memory fragmentation issues
            high_fragmentation = [
                (strategy, data["fragmentation_percentage"])
                for strategy, data in analysis["fragmentation_analysis"].items()
                if "fragmentation_percentage" in data and data["fragmentation_percentage"] > 10
            ]
            
            if high_fragmentation:
                for strategy, frag_pct in high_fragmentation:
                    analysis["recommendations"].append(
                        f"Strategy {strategy} shows significant memory fragmentation "
                        f"({frag_pct:.1f}%). This may impact long-running processes."
                    )
        
        return analysis
    
    def _generate_visualization_data(self, benchmark: QuantizationBenchmark) -> Dict[str, Any]:
        """
        Generate data structures for visualization.
        
        Args:
            benchmark: Benchmark results to visualize
            
        Returns:
            Dictionary with visualization data
        """
        visualization_data = {
            "memory_usage": [],
            "inference_speed": [],
            "accuracy": [],
            "efficiency_scores": [],
            "hardware_compatibility": [],
            "memory_breakdown": [],
            "memory_timeline": {},
            "tradeoff_analysis": [],
            "overhead_analysis": [],
            "fragmentation_analysis": []
        }
        
        # Filter valid results
        valid_results = [r for r in benchmark.results if r.error is None]
        if not valid_results:
            return visualization_data
        
        # Get strategies for ordering
        strategies = [r.strategy for r in valid_results]
        
        # Prepare data for each visualization type
        for result in valid_results:
            # Basic metrics
            visualization_data["memory_usage"].append({
                "strategy": result.strategy,
                "memory_gb": result.memory_used_gb
            })
            
            visualization_data["inference_speed"].append({
                "strategy": result.strategy,
                "time_per_image": result.inference_time
            })
            
            visualization_data["accuracy"].append({
                "strategy": result.strategy,
                "accuracy": result.accuracy
            })
            
            visualization_data["efficiency_scores"].append({
                "strategy": result.strategy,
                "score": result.efficiency_score
            })
            
            visualization_data["hardware_compatibility"].append({
                "strategy": result.strategy,
                "compatible": result.hardware_compatibility.get("compatible", True),
                "exceeds_memory": result.hardware_compatibility.get("exceeds_memory", False)
            })
            
            # Advanced visualization data - memory breakdown
            if hasattr(result, "detailed_stats") and "memory" in result.detailed_stats:
                memory_stats = result.detailed_stats.get("memory", {})
                
                visualization_data["memory_breakdown"].append({
                    "strategy": result.strategy,
                    "model_size": memory_stats.get("model_only", result.memory_used_gb),
                    "inference_overhead": memory_stats.get("inference_overhead", 0.0),
                    "retained_after_cleanup": memory_stats.get("after_cleanup", 0.0) - memory_stats.get("before_loading", 0.0)
                })
                
                # Prepare memory timeline data if available
                if "memory_timeline" in result.detailed_stats:
                    timeline = result.detailed_stats["memory_timeline"]
                    
                    visualization_data["memory_timeline"][result.strategy] = {
                        "timestamps": [p.get("timestamp", 0) for p in timeline],
                        "memory_values": [p.get("memory_gb", 0) for p in timeline],
                        "phases": [p.get("phase", "") for p in timeline]
                    }
        
        # Add trade-off analysis - accuracy vs memory vs speed
        for result in valid_results:
            visualization_data["tradeoff_analysis"].append({
                "strategy": result.strategy,
                "accuracy": result.accuracy,
                "memory_gb": result.memory_used_gb,
                "inference_time": result.inference_time,
                "efficiency_score": result.efficiency_score
            })
            
        # Include overhead analysis
        memory_analysis = self._generate_memory_analysis(benchmark)
        for strategy, data in memory_analysis.get("overhead_analysis", {}).items():
            visualization_data["overhead_analysis"].append({
                "strategy": strategy,
                "base_memory_gb": data.get("base_memory_gb", 0),
                "peak_memory_gb": data.get("peak_memory_gb", 0),
                "growth_percentage": data.get("growth_percentage", 0)
            })
            
        # Include fragmentation analysis
        for strategy, data in memory_analysis.get("fragmentation_analysis", {}).items():
            if "fragmentation_percentage" in data:
                visualization_data["fragmentation_analysis"].append({
                    "strategy": strategy,
                    "fragmentation_percentage": data.get("fragmentation_percentage", 0),
                    "retained_memory_gb": data.get("retained_memory_gb", 0)
                })
        
        # Add comparative analysis - normalize values for easier comparison
        visualization_data["normalized_comparison"] = self._generate_normalized_comparison(valid_results)
        
        # Add memory efficiency (accuracy per GB)
        visualization_data["memory_efficiency"] = []
        for strategy, data in memory_analysis.get("memory_efficiency", {}).items():
            visualization_data["memory_efficiency"].append({
                "strategy": strategy,
                "accuracy_per_gb": data.get("accuracy_per_gb", 0),
                "relative_efficiency": data.get("relative_efficiency", 0)
            })
        
        return visualization_data
    
    def _generate_normalized_comparison(self, results: List[QuantizationResult]) -> Dict[str, Any]:
        """
        Generate normalized data for comparative visualization.
        
        Args:
            results: List of quantization results
            
        Returns:
            Dictionary with normalized comparison data
        """
        if not results:
            return {}
            
        # Extract metrics to normalize
        strategies = [r.strategy for r in results]
        accuracies = [r.accuracy for r in results]
        memory_usages = [r.memory_used_gb for r in results]
        inference_times = [r.inference_time for r in results]
        
        # Find maximum values (for normalization)
        max_accuracy = max(accuracies) if accuracies else 1
        max_memory = max(memory_usages) if memory_usages else 1
        max_time = max(inference_times) if inference_times else 1
        
        # Create normalized data (0-1 scale)
        normalized_data = {
            "strategies": strategies,
            "metrics": [
                {
                    "name": "Accuracy",
                    "values": [a / max_accuracy if max_accuracy > 0 else 0 for a in accuracies],
                    "higher_is_better": True
                },
                {
                    "name": "Memory Usage",
                    "values": [m / max_memory if max_memory > 0 else 0 for m in memory_usages],
                    "higher_is_better": False
                },
                {
                    "name": "Inference Time",
                    "values": [t / max_time if max_time > 0 else 0 for t in inference_times],
                    "higher_is_better": False
                }
            ]
        }
        
        # Add inverse metrics where lower is better (for radar charts)
        normalized_data["inverted_metrics"] = {}
        for metric in normalized_data["metrics"]:
            if not metric["higher_is_better"]:
                # Invert values (1 - value) so higher is always better in radar charts
                normalized_data["inverted_metrics"][metric["name"]] = [
                    1 - v for v in metric["values"]
                ]
            else:
                normalized_data["inverted_metrics"][metric["name"]] = metric["values"]
        
        return normalized_data

# Create a global instance function for convenience
def get_quantization_analyzer(
    model_name: str,
    config: Optional[ExperimentConfiguration] = None,
    results_dir: Optional[Union[str, Path]] = None
) -> 'QuantizationAnalyzer':
    """
    Get or create a QuantizationAnalyzer instance.
    
    Args:
        model_name: Name of the model to analyze
        config: Optional experiment configuration
        results_dir: Directory to store results
        
    Returns:
        QuantizationAnalyzer instance
    """
    # Use a module-level singleton
    global _quantization_analyzer
    
    if '_quantization_analyzer' not in globals() or _quantization_analyzer is None:
        _quantization_analyzer = QuantizationAnalyzer(
            model_name=model_name,
            config=config,
            results_dir=results_dir
        )
    
    return _quantization_analyzer