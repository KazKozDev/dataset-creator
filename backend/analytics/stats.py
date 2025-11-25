"""
Dataset Statistics

Provides statistical analysis of datasets
"""

from typing import Dict, Any, List
from collections import Counter
import statistics


class DatasetStats:
    """Statistical analysis for datasets"""

    @staticmethod
    def analyze_dataset(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze a dataset and return comprehensive statistics

        Args:
            examples: List of dataset examples

        Returns:
            Dictionary containing various statistics
        """
        if not examples:
            return {"error": "No examples to analyze"}

        stats = {
            "basic": DatasetStats._get_basic_stats(examples),
            "content": DatasetStats._get_content_stats(examples),
            "diversity": DatasetStats._get_diversity_stats(examples),
            "quality_indicators": DatasetStats._get_quality_indicators(examples)
        }

        return stats

    @staticmethod
    def _get_basic_stats(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get basic statistics"""
        return {
            "total_examples": len(examples),
            "sample_fields": list(examples[0].keys()) if examples else []
        }

    @staticmethod
    def _get_content_stats(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get content-related statistics"""
        stats = {}

        # Analyze conversation lengths
        if any("conversation" in ex or "messages" in ex for ex in examples):
            turn_counts = []
            for ex in examples:
                conversation = ex.get("conversation", ex.get("messages", []))
                if conversation:
                    turn_counts.append(len(conversation))

            if turn_counts:
                stats["conversation_turns"] = {
                    "min": min(turn_counts),
                    "max": max(turn_counts),
                    "avg": round(statistics.mean(turn_counts), 2),
                    "median": statistics.median(turn_counts)
                }

        # Analyze text lengths
        text_lengths = []
        for ex in examples:
            # Try different text fields
            text = ""
            if "text" in ex:
                text = ex["text"]
            elif "content" in ex:
                text = ex["content"]
            elif "conversation" in ex:
                text = " ".join([msg.get("content", "") for msg in ex["conversation"]])
            elif "messages" in ex:
                text = " ".join([msg.get("content", "") for msg in ex["messages"]])
            elif "prompt" in ex and "completion" in ex:
                text = ex["prompt"] + " " + ex["completion"]

            text_lengths.append(len(text))

        if text_lengths:
            stats["text_length_chars"] = {
                "min": min(text_lengths),
                "max": max(text_lengths),
                "avg": round(statistics.mean(text_lengths), 2),
                "median": statistics.median(text_lengths)
            }

            # Estimate tokens (rough: ~4 chars per token)
            token_lengths = [length // 4 for length in text_lengths]
            stats["estimated_tokens"] = {
                "min": min(token_lengths),
                "max": max(token_lengths),
                "avg": round(statistics.mean(token_lengths), 2),
                "total": sum(token_lengths)
            }

        return stats

    @staticmethod
    def _get_diversity_stats(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get diversity-related statistics"""
        stats = {}

        # Domain/subdomain distribution
        if any("domain" in ex for ex in examples):
            domains = [ex.get("domain", "unknown") for ex in examples]
            domain_counts = Counter(domains)
            stats["domain_distribution"] = dict(domain_counts.most_common())

        if any("subdomain" in ex for ex in examples):
            subdomains = [ex.get("subdomain", "unknown") for ex in examples]
            subdomain_counts = Counter(subdomains)
            stats["subdomain_distribution"] = dict(subdomain_counts.most_common(10))

        # Language distribution
        if any("language" in ex for ex in examples):
            languages = [ex.get("language", "unknown") for ex in examples]
            lang_counts = Counter(languages)
            stats["language_distribution"] = dict(lang_counts.most_common())

        return stats

    @staticmethod
    def _get_quality_indicators(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get quality indicator statistics"""
        stats = {}

        # Quality scores if available
        if any("quality_score" in ex for ex in examples):
            scores = [ex.get("quality_score", 0) for ex in examples if "quality_score" in ex]
            if scores:
                stats["quality_scores"] = {
                    "min": min(scores),
                    "max": max(scores),
                    "avg": round(statistics.mean(scores), 2),
                    "median": statistics.median(scores)
                }

        # Flag statistics
        flagged_count = sum(1 for ex in examples if ex.get("flagged", False))
        if flagged_count > 0:
            stats["flagged_examples"] = {
                "count": flagged_count,
                "percentage": round(flagged_count / len(examples) * 100, 2)
            }

        # Metadata presence
        metadata_count = sum(1 for ex in examples if "metadata" in ex and ex["metadata"])
        stats["examples_with_metadata"] = {
            "count": metadata_count,
            "percentage": round(metadata_count / len(examples) * 100, 2)
        }

        return stats

    @staticmethod
    def compare_datasets(
        dataset1: List[Dict[str, Any]],
        dataset2: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compare two datasets

        Args:
            dataset1: First dataset
            dataset2: Second dataset

        Returns:
            Comparison results
        """
        stats1 = DatasetStats.analyze_dataset(dataset1)
        stats2 = DatasetStats.analyze_dataset(dataset2)

        comparison = {
            "dataset1": {
                "size": len(dataset1),
                "stats": stats1
            },
            "dataset2": {
                "size": len(dataset2),
                "stats": stats2
            },
            "size_difference": len(dataset2) - len(dataset1),
            "size_ratio": round(len(dataset2) / len(dataset1), 2) if len(dataset1) > 0 else 0
        }

        return comparison

    @staticmethod
    def get_recommendations(examples: List[Dict[str, Any]]) -> List[str]:
        """
        Get recommendations for improving the dataset

        Args:
            examples: List of dataset examples

        Returns:
            List of recommendations
        """
        recommendations = []

        if len(examples) < 100:
            recommendations.append(
                f"Dataset is small ({len(examples)} examples). "
                "Consider generating more examples for better model training."
            )

        # Check for quality scores
        has_quality_scores = any("quality_score" in ex for ex in examples)
        if not has_quality_scores:
            recommendations.append(
                "No quality scores found. Run quality control to assess example quality."
            )

        # Check for diversity
        if any("domain" in ex for ex in examples):
            domains = set(ex.get("domain") for ex in examples)
            if len(domains) == 1:
                recommendations.append(
                    "Dataset has only one domain. Consider adding more diverse examples."
                )

        # Check text lengths
        stats = DatasetStats._get_content_stats(examples)
        if "text_length_chars" in stats:
            avg_length = stats["text_length_chars"]["avg"]
            if avg_length < 100:
                recommendations.append(
                    f"Average text length is short ({avg_length} chars). "
                    "Consider generating more detailed examples."
                )

        # Check for metadata
        metadata_count = sum(1 for ex in examples if ex.get("metadata"))
        if metadata_count == 0:
            recommendations.append(
                "No metadata found. Consider adding metadata for better dataset organization."
            )

        if not recommendations:
            recommendations.append("Dataset looks good! No immediate improvements needed.")

        return recommendations
