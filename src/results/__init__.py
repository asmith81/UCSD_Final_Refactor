"""
Results module initialization
"""

# Core classes and functions for easy access
from .collector import ResultsCollector, get_results_collector
from .schema import ExperimentResult, PromptPerformance, IndividualExtractionResult, ExtractionStatus

__all__ = [
    'ResultsCollector',
    'get_results_collector',
    'ExperimentResult',
    'PromptPerformance',
    'IndividualExtractionResult', 
    'ExtractionStatus'
]