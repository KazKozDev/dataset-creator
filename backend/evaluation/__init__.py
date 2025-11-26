"""
Dataset Evaluation Module

Provides comprehensive dataset quality metrics including:
- Diversity metrics (Distinct-n, Self-BLEU)
- Text statistics (length distribution, vocabulary)
- Semantic clustering
- Language detection
"""

from .dataset_evaluator import DatasetEvaluator, EvaluationReport
from .diversity_metrics import DiversityMetrics, calculate_distinct_n, calculate_self_bleu
from .text_stats import TextStatistics, calculate_text_stats
from .semantic_analyzer import SemanticAnalyzer

__all__ = [
    "DatasetEvaluator",
    "EvaluationReport",
    "DiversityMetrics",
    "calculate_distinct_n",
    "calculate_self_bleu",
    "TextStatistics",
    "calculate_text_stats",
    "SemanticAnalyzer",
]
