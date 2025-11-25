"""
Quality Report Generation

Generates comprehensive quality reports in various formats
"""

import json
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

from .toxicity import ToxicityDetector, ToxicityResult
from .deduplication import DeduplicationChecker, DeduplicationResult
from .diversity import DiversityAnalyzer, DiversityMetrics
from .scoring import QualityScorer, QualityScore


class QualityReportGenerator:
    """Generator for comprehensive quality reports"""

    def __init__(self, output_dir: str = "./data/quality_reports"):
        """
        Initialize report generator

        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.toxicity_detector = ToxicityDetector()
        self.dedup_checker = DeduplicationChecker()
        self.diversity_analyzer = DiversityAnalyzer()
        self.quality_scorer = QualityScorer()

    def generate_full_report(
        self,
        examples: List[Dict],
        dataset_name: str,
        text_field: str = "text",
        save_to_file: bool = True
    ) -> Dict:
        """
        Generate comprehensive quality report

        Args:
            examples: List of examples to analyze
            dataset_name: Name of the dataset
            text_field: Field containing text
            save_to_file: Whether to save report to file

        Returns:
            Full report dictionary
        """
        timestamp = datetime.now()

        # Run all analyses
        print("Running toxicity detection...")
        toxicity_stats = self._analyze_toxicity(examples, text_field)

        print("Running deduplication check...")
        dedup_stats = self._analyze_deduplication(examples, text_field)

        print("Running diversity analysis...")
        diversity_stats = self._analyze_diversity(examples, text_field)

        print("Calculating quality score...")
        quality_stats = self._analyze_quality(examples, text_field)

        # Compile full report
        report = {
            "dataset_name": dataset_name,
            "timestamp": timestamp.isoformat(),
            "summary": {
                "total_examples": len(examples),
                "overall_score": quality_stats["overall_score"],
                "grade": quality_stats["grade"],
                "status": self._determine_status(quality_stats["overall_score"])
            },
            "toxicity": toxicity_stats,
            "deduplication": dedup_stats,
            "diversity": diversity_stats,
            "quality": quality_stats,
            "recommendations": self._compile_recommendations(
                toxicity_stats,
                dedup_stats,
                diversity_stats,
                quality_stats
            )
        }

        # Save to file if requested
        if save_to_file:
            self._save_report(report, dataset_name)

        return report

    def generate_text_report(
        self,
        examples: List[Dict],
        dataset_name: str,
        text_field: str = "text"
    ) -> str:
        """
        Generate human-readable text report

        Args:
            examples: List of examples
            dataset_name: Dataset name
            text_field: Field containing text

        Returns:
            Formatted text report
        """
        report_data = self.generate_full_report(
            examples, dataset_name, text_field, save_to_file=False
        )

        report = f"""
{'=' * 70}
COMPREHENSIVE QUALITY REPORT
{'=' * 70}

Dataset: {report_data['dataset_name']}
Generated: {report_data['timestamp']}
Total Examples: {report_data['summary']['total_examples']:,}

Overall Quality Score: {report_data['summary']['overall_score']}/100
Grade: {report_data['summary']['grade']}
Status: {report_data['summary']['status']}

{'=' * 70}
TOXICITY ANALYSIS
{'=' * 70}

Total Examples: {report_data['toxicity']['total']}
Toxic Examples: {report_data['toxicity']['toxic_count']} ({report_data['toxicity']['toxic_percentage']}%)
Clean Examples: {report_data['toxicity']['clean_count']}
Average Score: {report_data['toxicity']['average_score']}

"""

        if report_data['toxicity']['category_breakdown']:
            report += "Category Breakdown:\n"
            for cat, count in report_data['toxicity']['category_breakdown'].items():
                report += f"  • {cat}: {count}\n"

        report += f"""
{'=' * 70}
DEDUPLICATION ANALYSIS
{'=' * 70}

Total Examples: {report_data['deduplication']['total_examples']}
Unique Examples: {report_data['deduplication']['unique_examples']}
Duplicate Count: {report_data['deduplication']['duplicate_count']}
Duplication Rate: {report_data['deduplication']['duplication_rate']}%
Method: {report_data['deduplication']['method']}

{'=' * 70}
DIVERSITY ANALYSIS
{'=' * 70}

Vocabulary Size: {report_data['diversity']['vocabulary_size']:,} unique tokens
Lexical Diversity: {report_data['diversity']['lexical_diversity']}
Entropy: {report_data['diversity']['entropy']}
"""

        if report_data['diversity']['domain_diversity']:
            report += f"Domain Diversity: {report_data['diversity']['domain_diversity']}\n"

        report += f"""
{'=' * 70}
QUALITY COMPONENTS
{'=' * 70}

"""

        for component, score in report_data['quality']['component_scores'].items():
            bar = '█' * int(score / 5) + '░' * (20 - int(score / 5))
            report += f"{component.capitalize():15} [{bar}] {score:.1f}/100\n"

        if report_data['quality']['issues']:
            report += f"\n⚠ ISSUES DETECTED:\n"
            for issue in report_data['quality']['issues']:
                report += f"  • {issue}\n"

        if report_data['quality']['strengths']:
            report += f"\n✓ STRENGTHS:\n"
            for strength in report_data['quality']['strengths']:
                report += f"  • {strength}\n"

        report += f"\n{'=' * 70}\n"
        report += "RECOMMENDATIONS\n"
        report += f"{'=' * 70}\n\n"

        for i, rec in enumerate(report_data['recommendations'], 1):
            report += f"{i}. {rec}\n"

        report += f"\n{'=' * 70}\n"

        return report

    def _analyze_toxicity(self, examples: List[Dict], text_field: str) -> Dict:
        """Run toxicity analysis"""
        texts = [self._extract_text(ex, text_field) for ex in examples]
        results = self.toxicity_detector.detect_batch(texts)
        return self.toxicity_detector.get_statistics(results)

    def _analyze_deduplication(self, examples: List[Dict], text_field: str) -> Dict:
        """Run deduplication analysis"""
        result = self.dedup_checker.check_exact_duplicates(examples, text_field)
        return self.dedup_checker.get_statistics(result)

    def _analyze_diversity(self, examples: List[Dict], text_field: str) -> Dict:
        """Run diversity analysis"""
        metrics = self.diversity_analyzer.analyze(examples, text_field)
        return {
            "vocabulary_size": metrics.vocabulary_size,
            "unique_token_ratio": metrics.unique_token_ratio,
            "lexical_diversity": metrics.lexical_diversity,
            "entropy": metrics.entropy,
            "domain_diversity": metrics.domain_diversity,
            "subdomain_diversity": metrics.subdomain_diversity,
            "length_variance": metrics.length_variance,
            "recommendations": metrics.recommendations
        }

    def _analyze_quality(self, examples: List[Dict], text_field: str) -> Dict:
        """Run quality scoring"""
        score = self.quality_scorer.score_dataset(examples, text_field)
        return {
            "overall_score": score.overall_score,
            "component_scores": score.component_scores,
            "grade": score.grade,
            "issues": score.issues,
            "strengths": score.strengths,
            "recommendations": score.recommendations
        }

    def _extract_text(self, example: Dict, text_field: str) -> str:
        """Extract text from example"""
        if text_field in example:
            return str(example[text_field])

        for field in ["text", "content", "prompt", "instruction"]:
            if field in example:
                return str(example[field])

        return ""

    def _determine_status(self, score: float) -> str:
        """Determine dataset status from score"""
        if score >= 85:
            return "READY FOR PRODUCTION"
        elif score >= 70:
            return "NEEDS IMPROVEMENT"
        else:
            return "NOT READY"

    def _compile_recommendations(
        self,
        toxicity_stats: Dict,
        dedup_stats: Dict,
        diversity_stats: Dict,
        quality_stats: Dict
    ) -> List[str]:
        """Compile all recommendations"""
        recommendations = []

        # Priority 1: Critical issues
        if toxicity_stats['toxic_percentage'] > 5:
            recommendations.append(
                f"CRITICAL: Remove toxic content ({toxicity_stats['toxic_percentage']}% toxic rate)"
            )

        if "pii" in toxicity_stats.get('category_breakdown', {}):
            recommendations.append(
                "CRITICAL: Remove Personal Identifiable Information (PII)"
            )

        # Priority 2: Quality issues
        if dedup_stats['duplication_rate'] > 10:
            recommendations.append(
                f"HIGH: Remove duplicates ({dedup_stats['duplication_rate']:.1f}% duplication)"
            )

        if diversity_stats['lexical_diversity'] < 0.4:
            recommendations.append(
                "MEDIUM: Improve diversity - add more varied examples"
            )

        # Add specific quality recommendations
        recommendations.extend(quality_stats['recommendations'])

        # Remove duplicates and sort by priority
        unique_recommendations = list(dict.fromkeys(recommendations))

        return unique_recommendations

    def _save_report(self, report: Dict, dataset_name: str) -> str:
        """Save report to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{dataset_name}_quality_report_{timestamp}.json"
        filepath = self.output_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # Also save text version
        text_filename = f"{dataset_name}_quality_report_{timestamp}.txt"
        text_filepath = self.output_dir / text_filename

        # Generate text report
        text_report = self.generate_text_report(
            [],  # Empty - we already have the data
            dataset_name
        )

        # But we need to use existing data, so reconstruct from report
        # For now, just save JSON path
        with open(text_filepath, 'w', encoding='utf-8') as f:
            f.write(f"Quality report saved to: {filepath}\n")
            f.write(f"\nSummary:\n")
            f.write(f"  Overall Score: {report['summary']['overall_score']}/100\n")
            f.write(f"  Grade: {report['summary']['grade']}\n")
            f.write(f"  Status: {report['summary']['status']}\n")

        return str(filepath)

    def compare_reports(
        self,
        report1: Dict,
        report2: Dict
    ) -> Dict:
        """
        Compare two quality reports

        Args:
            report1: First report
            report2: Second report

        Returns:
            Comparison results
        """
        return {
            "datasets": {
                "dataset1": report1["dataset_name"],
                "dataset2": report2["dataset_name"]
            },
            "score_comparison": {
                "dataset1_score": report1["summary"]["overall_score"],
                "dataset2_score": report2["summary"]["overall_score"],
                "difference": report2["summary"]["overall_score"] - report1["summary"]["overall_score"],
                "winner": "dataset2" if report2["summary"]["overall_score"] > report1["summary"]["overall_score"] else "dataset1"
            },
            "component_differences": {
                component: {
                    "dataset1": report1["quality"]["component_scores"].get(component, 0),
                    "dataset2": report2["quality"]["component_scores"].get(component, 0),
                    "difference": report2["quality"]["component_scores"].get(component, 0) - report1["quality"]["component_scores"].get(component, 0)
                }
                for component in set(list(report1["quality"]["component_scores"].keys()) + list(report2["quality"]["component_scores"].keys()))
            }
        }
