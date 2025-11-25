"""
Toxicity Detection

Detects toxic, harmful, or inappropriate content in text
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ToxicityResult:
    """Result of toxicity detection"""
    is_toxic: bool
    toxicity_score: float  # 0.0 to 1.0
    toxic_categories: List[str]
    flagged_spans: List[Tuple[str, str]]  # (text, category)
    details: Dict[str, any]


class ToxicityDetector:
    """Detector for toxic and harmful content"""

    def __init__(self, sensitivity: str = "medium"):
        """
        Initialize toxicity detector

        Args:
            sensitivity: Detection sensitivity - "low", "medium", "high"
        """
        self.sensitivity = sensitivity
        self._load_patterns()

    def _load_patterns(self):
        """Load toxicity detection patterns"""

        # Profanity and offensive language (basic patterns)
        self.profanity_patterns = [
            r'\b(fuck|shit|damn|bitch|bastard|asshole|crap)\b',
            r'\b(idiot|moron|stupid|dumb)\b',
        ]

        # Hate speech and discriminatory language
        self.hate_speech_patterns = [
            r'\b(racist|sexist|homophobic|transphobic)\b',
            r'\b(hate|despise|detest)\s+(women|men|gay|trans|black|white|asian)',
        ]

        # Violence and threats
        self.violence_patterns = [
            r'\b(kill|murder|assault|attack|harm|hurt|destroy)\b.*\b(you|them|him|her)\b',
            r'\b(threat|threaten|violence|violent)\b',
            r'\b(gun|knife|weapon|bomb)\b',
        ]

        # Sexual content
        self.sexual_patterns = [
            r'\b(sex|sexual|porn|pornography|explicit)\b',
            r'\b(nude|naked|nsfw)\b',
        ]

        # Self-harm and dangerous behavior
        self.self_harm_patterns = [
            r'\b(suicide|self-harm|self harm|kill myself)\b',
            r'\b(cut myself|hurt myself)\b',
        ]

        # Personal information (PII)
        self.pii_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{16}\b',  # Credit card
            r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b',  # Email
            r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',  # Phone
        ]

        # Category mappings
        self.pattern_categories = {
            "profanity": self.profanity_patterns,
            "hate_speech": self.hate_speech_patterns,
            "violence": self.violence_patterns,
            "sexual_content": self.sexual_patterns,
            "self_harm": self.self_harm_patterns,
            "pii": self.pii_patterns,
        }

        # Sensitivity thresholds
        self.thresholds = {
            "low": 0.7,
            "medium": 0.5,
            "high": 0.3,
        }

    def detect(self, text: str) -> ToxicityResult:
        """
        Detect toxicity in text

        Args:
            text: Text to analyze

        Returns:
            ToxicityResult with detection results
        """
        if not text or not text.strip():
            return ToxicityResult(
                is_toxic=False,
                toxicity_score=0.0,
                toxic_categories=[],
                flagged_spans=[],
                details={}
            )

        text_lower = text.lower()
        toxic_categories = []
        flagged_spans = []
        category_scores = {}

        # Check each category
        for category, patterns in self.pattern_categories.items():
            matches = []

            for pattern in patterns:
                found = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in found:
                    matches.append(match.group())
                    flagged_spans.append((match.group(), category))

            if matches:
                toxic_categories.append(category)
                category_scores[category] = len(matches)

        # Calculate overall toxicity score
        if not toxic_categories:
            toxicity_score = 0.0
        else:
            # Score based on number of categories and matches
            base_score = len(toxic_categories) / len(self.pattern_categories)
            total_matches = sum(category_scores.values())
            match_score = min(total_matches * 0.1, 0.5)  # Up to 0.5 from match count
            toxicity_score = min(base_score + match_score, 1.0)

        # Determine if toxic based on threshold
        threshold = self.thresholds.get(self.sensitivity, 0.5)
        is_toxic = toxicity_score >= threshold

        return ToxicityResult(
            is_toxic=is_toxic,
            toxicity_score=toxicity_score,
            toxic_categories=toxic_categories,
            flagged_spans=flagged_spans,
            details={
                "category_scores": category_scores,
                "threshold": threshold,
                "sensitivity": self.sensitivity
            }
        )

    def detect_batch(self, texts: List[str]) -> List[ToxicityResult]:
        """
        Detect toxicity in multiple texts

        Args:
            texts: List of texts to analyze

        Returns:
            List of ToxicityResult objects
        """
        return [self.detect(text) for text in texts]

    def filter_toxic(
        self,
        examples: List[Dict],
        text_field: str = "text"
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Filter examples into clean and toxic sets

        Args:
            examples: List of examples to filter
            text_field: Field containing text to check

        Returns:
            Tuple of (clean_examples, toxic_examples)
        """
        clean = []
        toxic = []

        for example in examples:
            text = self._extract_text(example, text_field)
            result = self.detect(text)

            if result.is_toxic:
                example["toxicity_result"] = {
                    "score": result.toxicity_score,
                    "categories": result.toxic_categories,
                    "flagged_count": len(result.flagged_spans)
                }
                toxic.append(example)
            else:
                clean.append(example)

        return clean, toxic

    def _extract_text(self, example: Dict, text_field: str) -> str:
        """Extract text from example based on field"""
        if text_field in example:
            return str(example[text_field])

        # Try common fields
        if "text" in example:
            return str(example["text"])
        elif "content" in example:
            return str(example["content"])
        elif "conversation" in example:
            # Join conversation
            messages = example["conversation"]
            return " ".join([msg.get("content", "") for msg in messages])
        elif "messages" in example:
            messages = example["messages"]
            return " ".join([msg.get("content", "") for msg in messages])
        elif "prompt" in example and "completion" in example:
            return f"{example['prompt']} {example['completion']}"

        return ""

    def get_statistics(self, results: List[ToxicityResult]) -> Dict:
        """
        Get statistics from multiple detection results

        Args:
            results: List of ToxicityResult objects

        Returns:
            Dictionary with statistics
        """
        if not results:
            return {"total": 0, "toxic_count": 0, "toxic_percentage": 0.0}

        toxic_count = sum(1 for r in results if r.is_toxic)
        avg_score = sum(r.toxicity_score for r in results) / len(results)

        # Category breakdown
        category_counts = {}
        for result in results:
            for category in result.toxic_categories:
                category_counts[category] = category_counts.get(category, 0) + 1

        return {
            "total": len(results),
            "toxic_count": toxic_count,
            "clean_count": len(results) - toxic_count,
            "toxic_percentage": round(toxic_count / len(results) * 100, 2),
            "average_score": round(avg_score, 3),
            "category_breakdown": category_counts
        }

    def create_report(self, results: List[ToxicityResult]) -> str:
        """
        Create a text report from detection results

        Args:
            results: List of ToxicityResult objects

        Returns:
            Formatted report string
        """
        stats = self.get_statistics(results)

        report = f"""Toxicity Detection Report
{'=' * 50}

Total Examples: {stats['total']}
Toxic Examples: {stats['toxic_count']} ({stats['toxic_percentage']}%)
Clean Examples: {stats['clean_count']}
Average Toxicity Score: {stats['average_score']}

Sensitivity: {self.sensitivity}

Category Breakdown:
"""

        for category, count in sorted(stats['category_breakdown'].items(), key=lambda x: x[1], reverse=True):
            percentage = round(count / stats['total'] * 100, 1)
            report += f"  - {category}: {count} ({percentage}%)\n"

        report += f"\nRecommendations:\n"

        if stats['toxic_percentage'] > 10:
            report += "  - High toxicity rate detected. Consider reviewing and cleaning the dataset.\n"
        elif stats['toxic_percentage'] > 5:
            report += "  - Moderate toxicity detected. Review flagged examples.\n"
        else:
            report += "  - Low toxicity rate. Dataset appears clean.\n"

        if "pii" in stats['category_breakdown']:
            report += "  - PII detected! Remove personal information before sharing.\n"

        return report
