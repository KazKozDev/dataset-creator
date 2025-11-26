"""
Dataset Evaluator - Main Entry Point

Combines all evaluation metrics into a comprehensive report.
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

from .diversity_metrics import analyze_diversity, DiversityMetrics
from .text_stats import calculate_text_stats, TextStatistics
from .semantic_analyzer import SemanticAnalyzer, SemanticAnalysis

logger = logging.getLogger(__name__)


@dataclass
class EvaluationReport:
    """Comprehensive dataset evaluation report"""
    dataset_id: Optional[int]
    dataset_name: str
    evaluated_at: str
    total_examples: int
    
    # Diversity metrics
    diversity: DiversityMetrics
    
    # Text statistics
    text_stats: TextStatistics
    
    # Semantic analysis
    semantic: SemanticAnalysis
    
    # Overall scores
    overall_diversity_score: float  # 0-1
    overall_quality_score: float  # 0-1
    
    # Recommendations
    recommendations: List[str]
    
    # Warnings
    warnings: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "evaluated_at": self.evaluated_at,
            "total_examples": self.total_examples,
            "diversity": asdict(self.diversity),
            "text_stats": {
                "total_examples": self.text_stats.total_examples,
                "total_tokens": self.text_stats.total_tokens,
                "total_characters": self.text_stats.total_characters,
                "question_length": asdict(self.text_stats.question_length),
                "answer_length": asdict(self.text_stats.answer_length),
                "avg_qa_ratio": self.text_stats.avg_qa_ratio,
                "vocabulary_size": self.text_stats.vocabulary_size,
                "top_words": self.text_stats.top_words,
                "languages": self.text_stats.languages,
                "has_code_blocks": self.text_stats.has_code_blocks,
                "has_lists": self.text_stats.has_lists,
                "has_urls": self.text_stats.has_urls,
                "has_numbers": self.text_stats.has_numbers,
                "empty_responses": self.text_stats.empty_responses,
                "very_short_responses": self.text_stats.very_short_responses,
                "very_long_responses": self.text_stats.very_long_responses,
            },
            "semantic": {
                "num_clusters": self.semantic.num_clusters,
                "semantic_diversity": self.semantic.semantic_diversity,
                "avg_cluster_size": self.semantic.avg_cluster_size,
                "largest_cluster_ratio": self.semantic.largest_cluster_ratio,
                "outlier_count": self.semantic.outlier_count,
                "topic_distribution": self.semantic.topic_distribution,
                "clusters": [
                    {
                        "cluster_id": c.cluster_id,
                        "size": c.size,
                        "centroid_text": c.centroid_text,
                        "keywords": c.keywords,
                    }
                    for c in self.semantic.clusters
                ]
            },
            "overall_diversity_score": self.overall_diversity_score,
            "overall_quality_score": self.overall_quality_score,
            "recommendations": self.recommendations,
            "warnings": self.warnings,
        }


class DatasetEvaluator:
    """
    Main evaluator class that combines all metrics.
    """
    
    def __init__(self, use_semantic: bool = True):
        """
        Initialize the evaluator.
        
        Args:
            use_semantic: Whether to perform semantic analysis (requires more resources)
        """
        self.use_semantic = use_semantic
        self.semantic_analyzer = SemanticAnalyzer() if use_semantic else None
    
    def evaluate(
        self,
        examples: List[Dict],
        dataset_id: Optional[int] = None,
        dataset_name: str = "Unknown"
    ) -> EvaluationReport:
        """
        Perform comprehensive evaluation of a dataset.
        
        Args:
            examples: List of examples in chat format
            dataset_id: Optional dataset ID
            dataset_name: Name of the dataset
        
        Returns:
            EvaluationReport with all metrics
        """
        logger.info(f"Evaluating dataset: {dataset_name} ({len(examples)} examples)")
        
        if not examples:
            return self._empty_report(dataset_id, dataset_name)
        
        # Extract texts for diversity analysis
        texts = self._extract_all_texts(examples)
        
        # Calculate diversity metrics
        logger.info("Calculating diversity metrics...")
        diversity = analyze_diversity(texts)
        
        # Calculate text statistics
        logger.info("Calculating text statistics...")
        text_stats = calculate_text_stats(examples)
        
        # Perform semantic analysis
        if self.use_semantic and self.semantic_analyzer:
            logger.info("Performing semantic analysis...")
            semantic = self.semantic_analyzer.analyze(examples)
        else:
            semantic = SemanticAnalysis(
                num_clusters=0,
                clusters=[],
                semantic_diversity=0.0,
                avg_cluster_size=0.0,
                largest_cluster_ratio=0.0,
                outlier_count=0,
                topic_distribution={}
            )
        
        # Calculate overall scores
        overall_diversity = self._calculate_overall_diversity(diversity, semantic)
        overall_quality = self._calculate_overall_quality(text_stats, diversity)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            diversity, text_stats, semantic, overall_diversity, overall_quality
        )
        
        # Generate warnings
        warnings = self._generate_warnings(text_stats, diversity)
        
        return EvaluationReport(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            evaluated_at=datetime.now().isoformat(),
            total_examples=len(examples),
            diversity=diversity,
            text_stats=text_stats,
            semantic=semantic,
            overall_diversity_score=overall_diversity,
            overall_quality_score=overall_quality,
            recommendations=recommendations,
            warnings=warnings
        )
    
    def _extract_all_texts(self, examples: List[Dict]) -> List[str]:
        """Extract all text content from examples"""
        texts = []
        for example in examples:
            if "messages" in example:
                for msg in example["messages"]:
                    content = msg.get("content", "")
                    if content:
                        texts.append(content)
            elif "conversation" in example:
                for msg in example["conversation"]:
                    content = msg.get("content", "")
                    if content:
                        texts.append(content)
            elif "text" in example:
                texts.append(example["text"])
            elif "prompt" in example and "completion" in example:
                texts.append(example["prompt"])
                texts.append(example["completion"])
        return texts
    
    def _calculate_overall_diversity(
        self,
        diversity: DiversityMetrics,
        semantic: SemanticAnalysis
    ) -> float:
        """Calculate overall diversity score"""
        # Combine multiple diversity signals
        scores = []
        
        # Distinct-n scores (higher is better)
        distinct_avg = (diversity.distinct_1 + diversity.distinct_2 + diversity.distinct_3) / 3
        scores.append(min(distinct_avg * 2, 1.0))  # Scale up, cap at 1
        
        # Self-BLEU (lower is better, so invert)
        scores.append(1 - diversity.self_bleu_avg)
        
        # Semantic diversity if available
        if semantic.semantic_diversity > 0:
            scores.append(semantic.semantic_diversity)
        
        # Vocabulary richness (normalized)
        if diversity.total_tokens > 0:
            vocab_ratio = diversity.vocabulary_size / diversity.total_tokens
            scores.append(min(vocab_ratio * 5, 1.0))  # Scale up
        
        return round(sum(scores) / len(scores), 4) if scores else 0.0
    
    def _calculate_overall_quality(
        self,
        text_stats: TextStatistics,
        diversity: DiversityMetrics
    ) -> float:
        """Calculate overall quality score"""
        if text_stats.total_examples == 0:
            return 0.0
        
        scores = []
        total = text_stats.total_examples
        
        # Penalize empty responses
        empty_ratio = text_stats.empty_responses / total
        scores.append(1 - empty_ratio)
        
        # Penalize very short responses
        short_ratio = text_stats.very_short_responses / total
        scores.append(1 - min(short_ratio, 0.5) * 2)  # Cap penalty
        
        # Reward good Q/A ratio (answers should be longer than questions)
        if text_stats.avg_qa_ratio > 0:
            qa_score = min(text_stats.avg_qa_ratio / 3, 1.0)  # Ideal: 3x longer
            scores.append(qa_score)
        
        # Diversity contributes to quality
        scores.append(diversity.diversity_score)
        
        # Vocabulary richness
        if diversity.total_tokens > 0:
            vocab_ratio = diversity.vocabulary_size / diversity.total_tokens
            scores.append(min(vocab_ratio * 3, 1.0))
        
        return round(sum(scores) / len(scores), 4) if scores else 0.0
    
    def _generate_recommendations(
        self,
        diversity: DiversityMetrics,
        text_stats: TextStatistics,
        semantic: SemanticAnalysis,
        overall_diversity: float,
        overall_quality: float
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Diversity recommendations
        if diversity.self_bleu_avg > 0.5:
            recommendations.append(
                "High Self-BLEU score indicates similar examples. "
                "Consider adding more varied content or rephrasing existing examples."
            )
        
        if diversity.distinct_1 < 0.3:
            recommendations.append(
                "Low Distinct-1 score. Dataset uses repetitive vocabulary. "
                "Try using synonyms and varied phrasing."
            )
        
        # Text stats recommendations
        if text_stats.very_short_responses > text_stats.total_examples * 0.2:
            recommendations.append(
                f"{text_stats.very_short_responses} examples have very short responses (<10 words). "
                "Consider expanding these with more detailed answers."
            )
        
        if text_stats.avg_qa_ratio < 1.5:
            recommendations.append(
                "Answers are relatively short compared to questions. "
                "For training, longer, more detailed responses often work better."
            )
        
        # Semantic recommendations
        if semantic.largest_cluster_ratio > 0.4:
            recommendations.append(
                f"One topic dominates {semantic.largest_cluster_ratio*100:.0f}% of the dataset. "
                "Consider adding examples from other topics for better balance."
            )
        
        if semantic.num_clusters < 3 and text_stats.total_examples > 50:
            recommendations.append(
                "Dataset has low topic diversity. "
                "Consider adding examples covering different subjects or use cases."
            )
        
        # Overall recommendations
        if overall_diversity < 0.4:
            recommendations.append(
                "Overall diversity is low. Use data augmentation or add more varied examples."
            )
        
        if overall_quality < 0.5:
            recommendations.append(
                "Overall quality score is below average. "
                "Review and improve low-quality examples."
            )
        
        if not recommendations:
            recommendations.append(
                "Dataset shows good quality and diversity. Ready for training!"
            )
        
        return recommendations
    
    def _generate_warnings(
        self,
        text_stats: TextStatistics,
        diversity: DiversityMetrics
    ) -> List[str]:
        """Generate warnings for potential issues"""
        warnings = []
        
        if text_stats.empty_responses > 0:
            warnings.append(
                f"⚠️ {text_stats.empty_responses} examples have empty responses"
            )
        
        if text_stats.total_examples < 100:
            warnings.append(
                "⚠️ Dataset is small (<100 examples). May not be sufficient for fine-tuning."
            )
        
        if diversity.vocabulary_size < 500:
            warnings.append(
                "⚠️ Very limited vocabulary. Dataset may be too narrow."
            )
        
        if "mixed" in text_stats.languages:
            warnings.append(
                "⚠️ Dataset contains mixed languages. Ensure this is intentional."
            )
        
        return warnings
    
    def _empty_report(
        self,
        dataset_id: Optional[int],
        dataset_name: str
    ) -> EvaluationReport:
        """Return empty report for empty dataset"""
        from .diversity_metrics import DiversityMetrics
        from .text_stats import TextStatistics, LengthDistribution
        from .semantic_analyzer import SemanticAnalysis
        
        empty_length = LengthDistribution(
            min_length=0, max_length=0, mean_length=0.0, median_length=0.0,
            std_dev=0.0, percentile_25=0.0, percentile_75=0.0, percentile_90=0.0,
            histogram={}
        )
        
        return EvaluationReport(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            evaluated_at=datetime.now().isoformat(),
            total_examples=0,
            diversity=DiversityMetrics(
                distinct_1=0.0, distinct_2=0.0, distinct_3=0.0,
                self_bleu_1=0.0, self_bleu_2=0.0, self_bleu_3=0.0,
                self_bleu_avg=0.0, vocabulary_size=0, total_tokens=0,
                diversity_score=0.0
            ),
            text_stats=TextStatistics(
                total_examples=0, total_tokens=0, total_characters=0,
                question_length=empty_length, answer_length=empty_length,
                avg_qa_ratio=0.0, vocabulary_size=0, top_words=[],
                languages={}, has_code_blocks=0, has_lists=0,
                has_urls=0, has_numbers=0, empty_responses=0,
                very_short_responses=0, very_long_responses=0
            ),
            semantic=SemanticAnalysis(
                num_clusters=0, clusters=[], semantic_diversity=0.0,
                avg_cluster_size=0.0, largest_cluster_ratio=0.0,
                outlier_count=0, topic_distribution={}
            ),
            overall_diversity_score=0.0,
            overall_quality_score=0.0,
            recommendations=["Dataset is empty. Add examples to evaluate."],
            warnings=["⚠️ Dataset is empty"]
        )
