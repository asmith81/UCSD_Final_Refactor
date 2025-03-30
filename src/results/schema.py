"""
Results Schema Definition

This module defines a standardized, type-safe schema for storing 
multi-field extraction experiment results.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
from enum import Enum, auto

class ExtractionStatus(Enum):
    """Possible status of an extraction attempt."""
    SUCCESS = auto()
    PARTIAL_MATCH = auto()
    NO_MATCH = auto()
    ERROR = auto()

@dataclass
class IndividualExtractionResult:
    """
    Represents the result of extracting a specific field from a single document.
    """
    image_id: str
    field: str
    ground_truth: Optional[str] = None
    extracted_value: Optional[str] = None
    
    # Performance metrics
    exact_match: bool = False
    character_error_rate: float = 1.0
    confidence_score: float = 0.0
    
    # Extraction details
    status: ExtractionStatus = ExtractionStatus.ERROR
    processing_time: float = 0.0
    
    # Error handling
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert extraction result to a dictionary."""
        return {
            "image_id": self.image_id,
            "field": self.field,
            "ground_truth": self.ground_truth,
            "extracted_value": self.extracted_value,
            "exact_match": self.exact_match,
            "character_error_rate": self.character_error_rate,
            "confidence_score": self.confidence_score,
            "status": self.status.name,
            "processing_time": self.processing_time,
            "error_message": self.error_message
        }

@dataclass
class PromptPerformance:
    """
    Performance metrics for a specific prompt on a particular field.
    """
    prompt_name: str
    field: str
    
    # Aggregate metrics
    total_items: int = 0
    successful_extractions: int = 0
    accuracy: float = 0.0
    avg_character_error_rate: float = 1.0
    avg_processing_time: float = 0.0
    
    # Individual result tracking
    results: List[IndividualExtractionResult] = field(default_factory=list)
    
    def calculate_metrics(self):
        """
        Calculate performance metrics based on extraction results.
        """
        self.total_items = len(self.results)
        
        if self.total_items > 0:
            self.successful_extractions = sum(
                1 for result in self.results 
                if result.status in [ExtractionStatus.SUCCESS, ExtractionStatus.PARTIAL_MATCH]
            )
            
            self.accuracy = self.successful_extractions / self.total_items
            
            self.avg_character_error_rate = sum(
                result.character_error_rate for result in self.results
            ) / self.total_items
            
            self.avg_processing_time = sum(
                result.processing_time for result in self.results
            ) / self.total_items
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert prompt performance to a dictionary."""
        return {
            "prompt_name": self.prompt_name,
            "field": self.field,
            "total_items": self.total_items,
            "successful_extractions": self.successful_extractions,
            "accuracy": self.accuracy,
            "avg_character_error_rate": self.avg_character_error_rate,
            "avg_processing_time": self.avg_processing_time,
            "results": [result.to_dict() for result in self.results]
        }

@dataclass
class FieldResult:
    """
    Aggregate results for a specific field across multiple prompts.
    
    This class encapsulates all extraction results for a single field,
    organized by prompt, and provides aggregate metrics across all prompts.
    """
    field_name: str
    average_accuracy: float = 0.0
    sample_count: int = 0
    # Map of prompt name to its performance metrics
    prompt_performances: Dict[str, PromptPerformance] = field(default_factory=dict)
    
    # Additional field-specific metrics
    exact_match: Optional[float] = None
    character_error_rate: Optional[float] = None
    processing_time: Optional[float] = None
    
    def add_prompt_performance(self, prompt_performance: PromptPerformance) -> None:
        """
        Add a prompt's performance results to this field.
        
        Args:
            prompt_performance: Performance metrics for a specific prompt
        """
        self.prompt_performances[prompt_performance.prompt_name] = prompt_performance
        self._recalculate_metrics()
    
    def _recalculate_metrics(self) -> None:
        """
        Recalculate aggregate field metrics based on all prompt performances.
        """
        if not self.prompt_performances:
            self.average_accuracy = 0.0
            self.sample_count = 0
            return
            
        # Calculate weighted average across all prompts
        total_accuracy_weighted = 0.0
        total_samples = 0
        
        # Track other metrics
        all_exact_match = []
        all_cer = []
        all_times = []
        
        for prompt_perf in self.prompt_performances.values():
            samples = prompt_perf.total_items
            if samples > 0:
                total_accuracy_weighted += prompt_perf.accuracy * samples
                total_samples += samples
                
                if hasattr(prompt_perf, 'exact_match') and prompt_perf.exact_match is not None:
                    all_exact_match.append(prompt_perf.exact_match)
                    
                if hasattr(prompt_perf, 'character_error_rate') and prompt_perf.character_error_rate is not None:
                    all_cer.append(prompt_perf.character_error_rate)
                    
                if hasattr(prompt_perf, 'processing_time') and prompt_perf.processing_time is not None:
                    all_times.append(prompt_perf.processing_time)
        
        # Set aggregate metrics
        self.average_accuracy = total_accuracy_weighted / total_samples if total_samples > 0 else 0.0
        self.sample_count = total_samples
        
        # Set additional metrics if available
        if all_exact_match:
            self.exact_match = sum(all_exact_match) / len(all_exact_match)
            
        if all_cer:
            self.character_error_rate = sum(all_cer) / len(all_cer)
            
        if all_times:
            self.processing_time = sum(all_times) / len(all_times)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert field result to a dictionary."""
        return {
            "field_name": self.field_name,
            "average_accuracy": self.average_accuracy,
            "sample_count": self.sample_count,
            "exact_match": self.exact_match,
            "character_error_rate": self.character_error_rate,
            "processing_time": self.processing_time,
            "prompt_performances": {
                name: perf.to_dict() for name, perf in self.prompt_performances.items()
            }
        }
        
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'FieldResult':
        """
        Create a FieldResult instance from a dictionary.
        
        Args:
            data: Dictionary containing field result data
            
        Returns:
            FieldResult instance populated from the dictionary
        """
        field_result = FieldResult(
            field_name=data.get('field_name', ''),
            average_accuracy=data.get('average_accuracy', 0.0),
            sample_count=data.get('sample_count', 0),
            exact_match=data.get('exact_match'),
            character_error_rate=data.get('character_error_rate'),
            processing_time=data.get('processing_time')
        )
        
        # Populate prompt performances
        prompt_performances = data.get('prompt_performances', {})
        for prompt_name, prompt_data in prompt_performances.items():
            prompt_perf = PromptPerformance(
                prompt_name=prompt_name,
                field=field_result.field_name
            )
            
            # Copy metrics
            for metric in ['accuracy', 'exact_match', 'character_error_rate', 'processing_time']:
                if metric in prompt_data:
                    setattr(prompt_perf, metric, prompt_data[metric])
            
            prompt_perf.total_items = prompt_data.get('total_items', 0)
            prompt_perf.successful_extractions = prompt_data.get('successful_extractions', 0)
            
            field_result.prompt_performances[prompt_name] = prompt_perf
            
        return field_result

@dataclass
class ExperimentResult:
    """
    Comprehensive schema for storing entire experiment results.
    """
    experiment_name: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Experiment configuration
    model_name: Optional[str] = None
    prompt_strategy: Optional[str] = None
    
    # Field-specific results
    field_results: Dict[str, List[PromptPerformance]] = field(default_factory=dict)
    
    # Experiment-level metrics
    total_fields: int = 0
    total_items: int = 0
    overall_accuracy: float = 0.0
    
    def add_field_results(self, field: str, prompt_performance: PromptPerformance):
        """
        Add results for a specific field.
        
        Args:
            field: Name of the field
            prompt_performance: Performance results for the field
        """
        if field not in self.field_results:
            self.field_results[field] = []
        
        self.field_results[field].append(prompt_performance)
        
        # Recalculate experiment-level metrics
        self._calculate_experiment_metrics()
    
    def _calculate_experiment_metrics(self):
        """
        Calculate overall experiment-level metrics.
        """
        self.total_fields = len(self.field_results)
        
        # Calculate total items and overall accuracy
        total_items_across_fields = 0
        total_successful_extractions = 0
        
        for field_performances in self.field_results.values():
            for prompt_performance in field_performances:
                total_items_across_fields += prompt_performance.total_items
                total_successful_extractions += prompt_performance.successful_extractions
        
        self.total_items = total_items_across_fields
        self.overall_accuracy = (
            total_successful_extractions / total_items_across_fields 
            if total_items_across_fields > 0 else 0.0
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert experiment result to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the experiment result
        """
        return {
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp.isoformat(),
            "model_name": self.model_name,
            "prompt_strategy": self.prompt_strategy,
            "total_fields": self.total_fields,
            "total_items": self.total_items,
            "overall_accuracy": self.overall_accuracy,
            "field_results": {
                field: [perf.to_dict() for perf in performances]
                for field, performances in self.field_results.items()
            }
        }
    
    def save_to_file(self, filepath: str):
        """
        Save experiment results to a JSON file.
        
        Args:
            filepath: Path to save the results
        """
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'ExperimentResult':
        """
        Load experiment results from a JSON file.
        
        Args:
            filepath: Path to load results from
        
        Returns:
            Reconstructed ExperimentResult instance
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Create instance and populate
        experiment = cls(
            experiment_name=data['experiment_name'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            model_name=data.get('model_name'),
            prompt_strategy=data.get('prompt_strategy')
        )
        
        # Reconstruct field results
        for field, prompt_performances in data.get('field_results', {}).items():
            for perf_data in prompt_performances:
                prompt_perf = PromptPerformance(
                    prompt_name=perf_data['prompt_name'],
                    field=field
                )
                
                # Populate results
                prompt_perf.results = [
                    IndividualExtractionResult(
                        image_id=result['image_id'],
                        field=field,
                        ground_truth=result.get('ground_truth'),
                        extracted_value=result.get('extracted_value'),
                        exact_match=result.get('exact_match', False),
                        character_error_rate=result.get('character_error_rate', 1.0),
                        confidence_score=result.get('confidence_score', 0.0),
                        status=ExtractionStatus[result.get('status', 'ERROR')],
                        processing_time=result.get('processing_time', 0.0),
                        error_message=result.get('error_message')
                    )
                    for result in perf_data.get('results', [])
                ]
                
                # Recalculate metrics
                prompt_perf.calculate_metrics()
                
                # Add to experiment results
                experiment.add_field_results(field, prompt_perf)
        
        return experiment