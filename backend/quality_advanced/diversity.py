"""
Diversity Analysis

Analyzes diversity and variety in datasets
"""

import statistics
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from collections import Counter
import math


@dataclass
class DiversityMetrics:
    """Diversity analysis metrics"""
    vocabulary_size: int
    unique_token_ratio: float
    lexical_diversity: float  # Type-Token Ratio
    entropy: float
    domain_diversity: Optional[float]
    subdomain_diversity: Optional[float]
    length_variance: float
    topic_coverage: Dict[str, int]
    recommendations: List[str]


class DiversityAnalyzer:
    """Analyzer for dataset diversity"""

    def analyze(
        self,
        examples: List[Dict],
        text_field: str = "text"
    ) -> DiversityMetrics:
        """
        Analyze diversity of a dataset

        Args:
            examples: List of examples to analyze
            text_field: Field containing text

        Returns:
            DiversityMetrics with analysis results
        """
        if not examples:
            return self._empty_metrics()

        # Extract texts
        texts = [self._extract_text(ex, text_field) for ex in examples]

        # Calculate lexical diversity
        all_tokens = []
        for text in texts:
            all_tokens.extend(self._tokenize(text))

        total_tokens = len(all_tokens)
        unique_tokens = len(set(all_tokens))
        vocabulary_size = unique_tokens
        unique_token_ratio = unique_tokens / total_tokens if total_tokens > 0 else 0

        # Type-Token Ratio (TTR)
        # For large texts, use root TTR or moving average TTR
        if total_tokens > 1000:
            lexical_diversity = unique_tokens / math.sqrt(total_tokens)
        else:
            lexical_diversity = unique_tokens / total_tokens if total_tokens > 0 else 0

        # Calculate entropy (information content)
        token_counts = Counter(all_tokens)
        entropy = self._calculate_entropy(token_counts, total_tokens)

        # Domain and subdomain diversity
        domain_diversity = self._calculate_categorical_diversity(
            examples, "domain"
        )
        subdomain_diversity = self._calculate_categorical_diversity(
            examples, "subdomain"
        )

        # Length variance (measure of variety in text lengths)
        lengths = [len(text.split()) for text in texts]
        length_variance = statistics.variance(lengths) if len(lengths) > 1 else 0

        # Topic coverage (based on domains/subdomains)
        topic_coverage = self._analyze_topic_coverage(examples)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            vocabulary_size=vocabulary_size,
            lexical_diversity=lexical_diversity,
            entropy=entropy,
            domain_diversity=domain_diversity,
            length_variance=length_variance,
            topic_coverage=topic_coverage
        )

        return DiversityMetrics(
            vocabulary_size=vocabulary_size,
            unique_token_ratio=round(unique_token_ratio, 3),
            lexical_diversity=round(lexical_diversity, 3),
            entropy=round(entropy, 3),
            domain_diversity=round(domain_diversity, 3) if domain_diversity else None,
            subdomain_diversity=round(subdomain_diversity, 3) if subdomain_diversity else None,
            length_variance=round(length_variance, 2),
            topic_coverage=topic_coverage,
            recommendations=recommendations
        )

    def compare_diversity(
        self,
        dataset1: List[Dict],
        dataset2: List[Dict],
        text_field: str = "text"
    ) -> Dict:
        """
        Compare diversity between two datasets

        Args:
            dataset1: First dataset
            dataset2: Second dataset
            text_field: Field containing text

        Returns:
            Comparison results
        """
        metrics1 = self.analyze(dataset1, text_field)
        metrics2 = self.analyze(dataset2, text_field)

        return {
            "dataset1": {
                "vocabulary_size": metrics1.vocabulary_size,
                "lexical_diversity": metrics1.lexical_diversity,
                "entropy": metrics1.entropy
            },
            "dataset2": {
                "vocabulary_size": metrics2.vocabulary_size,
                "lexical_diversity": metrics2.lexical_diversity,
                "entropy": metrics2.entropy
            },
            "differences": {
                "vocabulary_diff": metrics2.vocabulary_size - metrics1.vocabulary_size,
                "diversity_diff": round(metrics2.lexical_diversity - metrics1.lexical_diversity, 3),
                "entropy_diff": round(metrics2.entropy - metrics1.entropy, 3)
            },
            "winner": self._determine_more_diverse(metrics1, metrics2)
        }

    def _extract_text(self, example: Dict, text_field: str) -> str:
        """Extract text from example"""
        if text_field in example:
            return str(example[text_field])

        if "text" in example:
            return str(example["text"])
        elif "content" in example:
            return str(example["content"])
        elif "conversation" in example:
            messages = example["conversation"]
            return " ".join([msg.get("content", "") for msg in messages])
        elif "messages" in example:
            messages = example["messages"]
            return " ".join([msg.get("content", "") for msg in messages])
        elif "prompt" in example and "completion" in example:
            return f"{example['prompt']} {example['completion']}"

        return ""

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        text = text.lower()
        # Simple word tokenization
        words = text.split()
        return [w.strip('.,!?;:()[]{}') for w in words if w.strip('.,!?;:()[]{}')]

    def _calculate_entropy(self, token_counts: Counter, total_tokens: int) -> float:
        """Calculate Shannon entropy"""
        if total_tokens == 0:
            return 0.0

        entropy = 0.0
        for count in token_counts.values():
            probability = count / total_tokens
            if probability > 0:
                entropy -= probability * math.log2(probability)

        return entropy

    def _calculate_categorical_diversity(
        self,
        examples: List[Dict],
        field: str
    ) -> Optional[float]:
        """
        Calculate diversity for categorical field using Simpson's diversity index

        Args:
            examples: List of examples
            field: Field name to analyze

        Returns:
            Diversity score (0-1) or None if field not present
        """
        values = [ex.get(field) for ex in examples if field in ex]

        if not values:
            return None

        # Count occurrences
        counts = Counter(values)
        total = len(values)

        # Simpson's Diversity Index: 1 - sum((n/N)^2)
        simpson = 1 - sum((count / total) ** 2 for count in counts.values())

        return simpson

    def _analyze_topic_coverage(self, examples: List[Dict]) -> Dict[str, int]:
        """Analyze topic coverage based on domains/subdomains"""
        coverage = {}

        # Count domains
        domains = [ex.get("domain") for ex in examples if "domain" in ex]
        if domains:
            domain_counts = Counter(domains)
            coverage["domains"] = dict(domain_counts.most_common(10))

        # Count subdomains
        subdomains = [ex.get("subdomain") for ex in examples if "subdomain" in ex]
        if subdomains:
            subdomain_counts = Counter(subdomains)
            coverage["subdomains"] = dict(subdomain_counts.most_common(10))

        return coverage

    def _generate_recommendations(
        self,
        vocabulary_size: int,
        lexical_diversity: float,
        entropy: float,
        domain_diversity: Optional[float],
        length_variance: float,
        topic_coverage: Dict
    ) -> List[str]:
        """Generate recommendations based on diversity metrics"""
        recommendations = []

        # Vocabulary recommendations
        if vocabulary_size < 1000:
            recommendations.append(
                "Low vocabulary size. Consider adding more diverse examples."
            )
        elif vocabulary_size > 50000:
            recommendations.append(
                "Very high vocabulary size. Dataset has excellent lexical diversity."
            )

        # Lexical diversity recommendations
        if lexical_diversity < 0.4:
            recommendations.append(
                "Low lexical diversity. Examples may be too similar or repetitive."
            )
        elif lexical_diversity > 0.7:
            recommendations.append(
                "High lexical diversity. Good variety in language use."
            )

        # Entropy recommendations
        if entropy < 8:
            recommendations.append(
                "Low entropy. Dataset may lack variety in word usage."
            )

        # Domain diversity recommendations
        if domain_diversity is not None:
            if domain_diversity < 0.3:
                recommendations.append(
                    "Low domain diversity. Consider adding examples from more domains."
                )
            elif domain_diversity > 0.7:
                recommendations.append(
                    "High domain diversity. Good coverage across domains."
                )

        # Length variance recommendations
        if length_variance < 10:
            recommendations.append(
                "Low length variance. Examples have similar lengths. Consider varying complexity."
            )

        # Topic coverage recommendations
        if topic_coverage:
            if "domains" in topic_coverage:
                domain_count = len(topic_coverage["domains"])
                if domain_count < 3:
                    recommendations.append(
                        f"Only {domain_count} domains represented. Add more for better coverage."
                    )

        if not recommendations:
            recommendations.append("Dataset shows good diversity across all metrics.")

        return recommendations

    def _determine_more_diverse(
        self,
        metrics1: DiversityMetrics,
        metrics2: DiversityMetrics
    ) -> str:
        """Determine which dataset is more diverse"""
        # Score based on multiple factors
        score1 = (
            metrics1.lexical_diversity * 0.4 +
            (metrics1.entropy / 15) * 0.3 +  # Normalize entropy
            (metrics1.domain_diversity or 0) * 0.3
        )

        score2 = (
            metrics2.lexical_diversity * 0.4 +
            (metrics2.entropy / 15) * 0.3 +
            (metrics2.domain_diversity or 0) * 0.3
        )

        if abs(score1 - score2) < 0.05:
            return "similar"
        elif score1 > score2:
            return "dataset1"
        else:
            return "dataset2"

    def _empty_metrics(self) -> DiversityMetrics:
        """Return empty metrics for empty dataset"""
        return DiversityMetrics(
            vocabulary_size=0,
            unique_token_ratio=0.0,
            lexical_diversity=0.0,
            entropy=0.0,
            domain_diversity=None,
            subdomain_diversity=None,
            length_variance=0.0,
            topic_coverage={},
            recommendations=["Dataset is empty"]
        )

    def create_report(self, metrics: DiversityMetrics) -> str:
        """Create a text report from diversity metrics"""
        report = f"""Diversity Analysis Report
{'=' * 50}

Lexical Metrics:
  Vocabulary Size: {metrics.vocabulary_size:,} unique tokens
  Unique Token Ratio: {metrics.unique_token_ratio}
  Lexical Diversity (TTR): {metrics.lexical_diversity}
  Entropy: {metrics.entropy}

Categorical Diversity:
"""

        if metrics.domain_diversity is not None:
            report += f"  Domain Diversity: {metrics.domain_diversity}\n"
        if metrics.subdomain_diversity is not None:
            report += f"  Subdomain Diversity: {metrics.subdomain_diversity}\n"

        report += f"\nVariance Metrics:\n"
        report += f"  Length Variance: {metrics.length_variance}\n"

        if metrics.topic_coverage:
            report += f"\nTopic Coverage:\n"
            if "domains" in metrics.topic_coverage:
                report += f"  Domains: {len(metrics.topic_coverage['domains'])}\n"
                for domain, count in list(metrics.topic_coverage['domains'].items())[:5]:
                    report += f"    - {domain}: {count} examples\n"

        report += f"\nRecommendations:\n"
        for rec in metrics.recommendations:
            report += f"  - {rec}\n"

        return report
