"""
Quality Scoring

Automated quality scoring system that combines multiple quality metrics
"""

from typing import Dict, List, Optional
from dataclasses import dataclass

from .toxicity import ToxicityDetector
from .deduplication import DeduplicationChecker
from .diversity import DiversityAnalyzer


@dataclass
class QualityScore:
    """Overall quality score for a dataset"""
    overall_score: float  # 0-100
    component_scores: Dict[str, float]
    grade: str  # A, B, C, D, F
    issues: List[str]
    strengths: List[str]
    recommendations: List[str]


class QualityScorer:
    """Automated quality scoring system"""

    def __init__(self):
        """Initialize quality scorer"""
        self.toxicity_detector = ToxicityDetector(sensitivity="medium")
        self.dedup_checker = DeduplicationChecker(similarity_threshold=0.90)
        self.diversity_analyzer = DiversityAnalyzer()

        # Scoring weights
        self.weights = {
            "toxicity": 0.30,  # 30% - Critical for safety
            "deduplication": 0.25,  # 25% - Important for quality
            "diversity": 0.25,  # 25% - Important for coverage
            "completeness": 0.10,  # 10% - Data completeness
            "consistency": 0.10   # 10% - Format consistency
        }

    def score_dataset(
        self,
        examples: List[Dict],
        text_field: str = "text"
    ) -> QualityScore:
        """
        Calculate comprehensive quality score for dataset

        Args:
            examples: List of examples to score
            text_field: Field containing text

        Returns:
            QualityScore with detailed breakdown
        """
        if not examples:
            return self._empty_score()

        component_scores = {}
        issues = []
        strengths = []

        # 1. Toxicity Score (higher is better - less toxic)
        toxicity_score, toxicity_issues, toxicity_strengths = self._score_toxicity(
            examples, text_field
        )
        component_scores["toxicity"] = toxicity_score
        issues.extend(toxicity_issues)
        strengths.extend(toxicity_strengths)

        # 2. Deduplication Score (higher is better - less duplicates)
        dedup_score, dedup_issues, dedup_strengths = self._score_deduplication(
            examples, text_field
        )
        component_scores["deduplication"] = dedup_score
        issues.extend(dedup_issues)
        strengths.extend(dedup_strengths)

        # 3. Diversity Score
        diversity_score, diversity_issues, diversity_strengths = self._score_diversity(
            examples, text_field
        )
        component_scores["diversity"] = diversity_score
        issues.extend(diversity_issues)
        strengths.extend(diversity_strengths)

        # 4. Completeness Score
        completeness_score, completeness_issues, completeness_strengths = self._score_completeness(
            examples
        )
        component_scores["completeness"] = completeness_score
        issues.extend(completeness_issues)
        strengths.extend(completeness_strengths)

        # 5. Consistency Score
        consistency_score, consistency_issues, consistency_strengths = self._score_consistency(
            examples
        )
        component_scores["consistency"] = consistency_score
        issues.extend(consistency_issues)
        strengths.extend(consistency_strengths)

        # Calculate overall score
        overall_score = sum(
            component_scores[component] * self.weights[component]
            for component in self.weights
        )

        # Determine grade
        grade = self._calculate_grade(overall_score)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            component_scores, issues, overall_score
        )

        return QualityScore(
            overall_score=round(overall_score, 1),
            component_scores={k: round(v, 1) for k, v in component_scores.items()},
            grade=grade,
            issues=issues,
            strengths=strengths,
            recommendations=recommendations
        )

    def _score_toxicity(
        self,
        examples: List[Dict],
        text_field: str
    ) -> tuple[float, List[str], List[str]]:
        """Score toxicity (0-100, higher is better)"""
        issues = []
        strengths = []

        # Detect toxicity
        texts = [self._extract_text(ex, text_field) for ex in examples]
        results = self.toxicity_detector.detect_batch(texts)
        stats = self.toxicity_detector.get_statistics(results)

        # Calculate score (inverse of toxicity rate)
        toxic_rate = stats['toxic_percentage'] / 100
        score = (1 - toxic_rate) * 100

        # Identify issues
        if toxic_rate > 0.1:
            issues.append(f"High toxicity rate: {stats['toxic_percentage']}%")
        elif toxic_rate > 0.05:
            issues.append(f"Moderate toxicity detected: {stats['toxic_percentage']}%")

        if "pii" in stats.get('category_breakdown', {}):
            issues.append("PII (Personal Identifiable Information) detected")

        # Identify strengths
        if toxic_rate < 0.01:
            strengths.append("Excellent toxicity control - dataset is very clean")
        elif toxic_rate < 0.05:
            strengths.append("Good toxicity control")

        return score, issues, strengths

    def _score_deduplication(
        self,
        examples: List[Dict],
        text_field: str
    ) -> tuple[float, List[str], List[str]]:
        """Score deduplication (0-100, higher is better)"""
        issues = []
        strengths = []

        # Check for duplicates
        result = self.dedup_checker.check_exact_duplicates(examples, text_field)
        dup_rate = result.duplication_rate

        # Calculate score (inverse of duplication rate)
        score = (1 - dup_rate) * 100

        # Identify issues
        if dup_rate > 0.2:
            issues.append(f"High duplication rate: {dup_rate * 100:.1f}%")
        elif dup_rate > 0.1:
            issues.append(f"Moderate duplication: {dup_rate * 100:.1f}%")

        # Identify strengths
        if dup_rate < 0.05:
            strengths.append("Excellent deduplication - minimal duplicates")
        elif dup_rate < 0.1:
            strengths.append("Good deduplication")

        return score, issues, strengths

    def _score_diversity(
        self,
        examples: List[Dict],
        text_field: str
    ) -> tuple[float, List[str], List[str]]:
        """Score diversity (0-100)"""
        issues = []
        strengths = []

        # Analyze diversity
        metrics = self.diversity_analyzer.analyze(examples, text_field)

        # Calculate score based on multiple factors
        lexical_score = min(metrics.lexical_diversity * 100, 100)
        entropy_score = min((metrics.entropy / 12) * 100, 100)  # Normalize to 100

        if metrics.domain_diversity is not None:
            domain_score = metrics.domain_diversity * 100
            score = (lexical_score * 0.4 + entropy_score * 0.3 + domain_score * 0.3)
        else:
            score = (lexical_score * 0.6 + entropy_score * 0.4)

        # Identify issues
        if metrics.lexical_diversity < 0.4:
            issues.append("Low lexical diversity - examples may be too similar")

        if metrics.vocabulary_size < 500:
            issues.append(f"Small vocabulary: {metrics.vocabulary_size} unique tokens")

        if metrics.domain_diversity and metrics.domain_diversity < 0.3:
            issues.append("Low domain diversity")

        # Identify strengths
        if metrics.lexical_diversity > 0.7:
            strengths.append("Excellent lexical diversity")

        if metrics.vocabulary_size > 10000:
            strengths.append(f"Rich vocabulary: {metrics.vocabulary_size:,} unique tokens")

        return score, issues, strengths

    def _score_completeness(self, examples: List[Dict]) -> tuple[float, List[str], List[str]]:
        """Score data completeness (0-100)"""
        issues = []
        strengths = []

        # Check for missing/empty fields
        empty_count = 0
        field_coverage = {}

        for example in examples:
            # Check if example has minimal data
            has_content = False

            for field in ["text", "content", "conversation", "messages", "prompt", "completion"]:
                if field in example and example[field]:
                    has_content = True
                    field_coverage[field] = field_coverage.get(field, 0) + 1

            if not has_content:
                empty_count += 1

        empty_rate = empty_count / len(examples) if examples else 0
        score = (1 - empty_rate) * 100

        # Check metadata completeness
        metadata_count = sum(1 for ex in examples if ex.get("metadata"))
        metadata_rate = metadata_count / len(examples) if examples else 0

        if metadata_rate > 0.8:
            score = min(score + 5, 100)  # Bonus for good metadata

        # Identify issues
        if empty_rate > 0.05:
            issues.append(f"{empty_count} examples with missing content")

        if metadata_rate < 0.5:
            issues.append(f"Low metadata coverage: {metadata_rate * 100:.1f}%")

        # Identify strengths
        if empty_rate == 0:
            strengths.append("Complete dataset - no empty examples")

        if metadata_rate > 0.8:
            strengths.append("Excellent metadata coverage")

        return score, issues, strengths

    def _score_consistency(self, examples: List[Dict]) -> tuple[float, List[str], List[str]]:
        """Score format consistency (0-100)"""
        issues = []
        strengths = []

        # Check format consistency
        formats = set()
        for example in examples:
            if "conversation" in example or "messages" in example:
                formats.add("conversation")
            elif "instruction" in example:
                formats.add("instruction")
            elif "prompt" in example:
                formats.add("prompt-completion")
            elif "text" in example or "content" in example:
                formats.add("text")

        format_count = len(formats)

        # Score based on format consistency
        if format_count == 1:
            score = 100
            strengths.append("Consistent format across all examples")
        elif format_count == 2:
            score = 80
        else:
            score = 60
            issues.append(f"Mixed formats detected: {format_count} different types")

        return score, issues, strengths

    def _extract_text(self, example: Dict, text_field: str) -> str:
        """Extract text from example"""
        if text_field in example:
            return str(example[text_field])

        for field in ["text", "content", "prompt", "instruction"]:
            if field in example:
                return str(example[field])

        return ""

    def _calculate_grade(self, score: float) -> str:
        """Calculate letter grade from score"""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"

    def _generate_recommendations(
        self,
        component_scores: Dict[str, float],
        issues: List[str],
        overall_score: float
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        # Toxicity recommendations
        if component_scores["toxicity"] < 80:
            recommendations.append(
                "Run toxicity filtering to remove harmful content"
            )

        # Deduplication recommendations
        if component_scores["deduplication"] < 80:
            recommendations.append(
                "Remove duplicate examples to improve dataset quality"
            )

        # Diversity recommendations
        if component_scores["diversity"] < 70:
            recommendations.append(
                "Add more diverse examples to improve coverage"
            )

        # Completeness recommendations
        if component_scores["completeness"] < 90:
            recommendations.append(
                "Fill in missing fields and add metadata"
            )

        # Consistency recommendations
        if component_scores["consistency"] < 80:
            recommendations.append(
                "Standardize format across all examples"
            )

        # Overall recommendations
        if overall_score < 70:
            recommendations.append(
                "Dataset needs significant quality improvements before production use"
            )
        elif overall_score < 85:
            recommendations.append(
                "Dataset is acceptable but has room for improvement"
            )

        if not recommendations:
            recommendations.append(
                "Dataset meets high quality standards. Ready for use!"
            )

        return recommendations

    def _empty_score(self) -> QualityScore:
        """Return empty score for empty dataset"""
        return QualityScore(
            overall_score=0.0,
            component_scores={},
            grade="F",
            issues=["Dataset is empty"],
            strengths=[],
            recommendations=["Add examples to the dataset"]
        )

    def create_report(self, score: QualityScore) -> str:
        """Create a detailed quality report"""
        report = f"""Dataset Quality Report
{'=' * 60}

Overall Score: {score.overall_score}/100 (Grade: {score.grade})

Component Scores:
"""

        for component, comp_score in sorted(score.component_scores.items(), key=lambda x: x[1]):
            bar = '█' * int(comp_score / 5) + '░' * (20 - int(comp_score / 5))
            report += f"  {component.capitalize():15} [{bar}] {comp_score:.1f}/100\n"

        if score.strengths:
            report += f"\n✓ Strengths:\n"
            for strength in score.strengths:
                report += f"  • {strength}\n"

        if score.issues:
            report += f"\n⚠ Issues:\n"
            for issue in score.issues:
                report += f"  • {issue}\n"

        report += f"\n📋 Recommendations:\n"
        for i, rec in enumerate(score.recommendations, 1):
            report += f"  {i}. {rec}\n"

        report += f"\n{'=' * 60}\n"

        if score.overall_score >= 85:
            report += "Status: ✓ READY FOR PRODUCTION\n"
        elif score.overall_score >= 70:
            report += "Status: ⚠ NEEDS IMPROVEMENT\n"
        else:
            report += "Status: ✗ NOT READY - REQUIRES SIGNIFICANT WORK\n"

        return report
