"""
Deduplication

Detects and removes duplicate examples from datasets
"""

import hashlib
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class DeduplicationResult:
    """Result of deduplication analysis"""
    total_examples: int
    unique_examples: int
    duplicate_count: int
    duplication_rate: float
    duplicate_groups: List[List[int]]  # Groups of duplicate indices
    similarity_threshold: float
    method: str


class DeduplicationChecker:
    """Checker for duplicate content in datasets"""

    def __init__(self, similarity_threshold: float = 0.95):
        """
        Initialize deduplication checker

        Args:
            similarity_threshold: Threshold for semantic similarity (0.0-1.0)
        """
        self.similarity_threshold = similarity_threshold

    def check_exact_duplicates(
        self,
        examples: List[Dict],
        text_field: str = "text"
    ) -> DeduplicationResult:
        """
        Check for exact duplicates using hash-based comparison

        Args:
            examples: List of examples to check
            text_field: Field containing text to compare

        Returns:
            DeduplicationResult with findings
        """
        hash_to_indices = defaultdict(list)

        # Hash each example
        for idx, example in enumerate(examples):
            text = self._extract_text(example, text_field)
            text_hash = self._hash_text(text)
            hash_to_indices[text_hash].append(idx)

        # Find duplicate groups
        duplicate_groups = [
            indices for indices in hash_to_indices.values()
            if len(indices) > 1
        ]

        # Count unique examples
        unique_count = len(hash_to_indices)
        duplicate_count = len(examples) - unique_count

        return DeduplicationResult(
            total_examples=len(examples),
            unique_examples=unique_count,
            duplicate_count=duplicate_count,
            duplication_rate=duplicate_count / len(examples) if examples else 0.0,
            duplicate_groups=duplicate_groups,
            similarity_threshold=1.0,  # Exact match
            method="exact_hash"
        )

    def check_fuzzy_duplicates(
        self,
        examples: List[Dict],
        text_field: str = "text"
    ) -> DeduplicationResult:
        """
        Check for fuzzy duplicates using normalized text comparison

        Args:
            examples: List of examples to check
            text_field: Field containing text to compare

        Returns:
            DeduplicationResult with findings
        """
        normalized_to_indices = defaultdict(list)

        # Normalize and group examples
        for idx, example in enumerate(examples):
            text = self._extract_text(example, text_field)
            normalized = self._normalize_text(text)
            normalized_to_indices[normalized].append(idx)

        # Find duplicate groups
        duplicate_groups = [
            indices for indices in normalized_to_indices.values()
            if len(indices) > 1
        ]

        unique_count = len(normalized_to_indices)
        duplicate_count = len(examples) - unique_count

        return DeduplicationResult(
            total_examples=len(examples),
            unique_examples=unique_count,
            duplicate_count=duplicate_count,
            duplication_rate=duplicate_count / len(examples) if examples else 0.0,
            duplicate_groups=duplicate_groups,
            similarity_threshold=0.95,
            method="fuzzy_normalized"
        )

    def check_semantic_duplicates(
        self,
        examples: List[Dict],
        text_field: str = "text"
    ) -> DeduplicationResult:
        """
        Check for semantic duplicates using simple similarity metrics

        Args:
            examples: List of examples to check
            text_field: Field containing text to compare

        Returns:
            DeduplicationResult with findings
        """
        # Extract texts
        texts = [self._extract_text(ex, text_field) for ex in examples]

        # Find similar pairs using Jaccard similarity
        duplicate_groups = []
        processed = set()

        for i in range(len(texts)):
            if i in processed:
                continue

            group = [i]
            tokens_i = set(self._tokenize(texts[i]))

            for j in range(i + 1, len(texts)):
                if j in processed:
                    continue

                tokens_j = set(self._tokenize(texts[j]))
                similarity = self._jaccard_similarity(tokens_i, tokens_j)

                if similarity >= self.similarity_threshold:
                    group.append(j)
                    processed.add(j)

            if len(group) > 1:
                duplicate_groups.append(group)
                processed.update(group)

        unique_count = len(examples) - sum(len(group) - 1 for group in duplicate_groups)
        duplicate_count = len(examples) - unique_count

        return DeduplicationResult(
            total_examples=len(examples),
            unique_examples=unique_count,
            duplicate_count=duplicate_count,
            duplication_rate=duplicate_count / len(examples) if examples else 0.0,
            duplicate_groups=duplicate_groups,
            similarity_threshold=self.similarity_threshold,
            method="semantic_jaccard"
        )

    def remove_duplicates(
        self,
        examples: List[Dict],
        method: str = "exact",
        text_field: str = "text"
    ) -> Tuple[List[Dict], List[List[Dict]]]:
        """
        Remove duplicates from examples

        Args:
            examples: List of examples
            method: Deduplication method - "exact", "fuzzy", "semantic"
            text_field: Field containing text to compare

        Returns:
            Tuple of (unique_examples, duplicate_groups)
        """
        # Get deduplication result
        if method == "exact":
            result = self.check_exact_duplicates(examples, text_field)
        elif method == "fuzzy":
            result = self.check_fuzzy_duplicates(examples, text_field)
        elif method == "semantic":
            result = self.check_semantic_duplicates(examples, text_field)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Build set of indices to keep (first in each group)
        indices_to_remove = set()
        for group in result.duplicate_groups:
            # Keep first, remove rest
            indices_to_remove.update(group[1:])

        # Extract unique examples
        unique_examples = [
            ex for idx, ex in enumerate(examples)
            if idx not in indices_to_remove
        ]

        # Build duplicate groups
        duplicate_groups = [
            [examples[idx] for idx in group]
            for group in result.duplicate_groups
        ]

        return unique_examples, duplicate_groups

    def _extract_text(self, example: Dict, text_field: str) -> str:
        """Extract text from example"""
        if text_field in example:
            return str(example[text_field])

        # Try common fields
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

    def _hash_text(self, text: str) -> str:
        """Create hash of text for exact comparison"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def _normalize_text(self, text: str) -> str:
        """Normalize text for fuzzy comparison"""
        # Convert to lowercase
        text = text.lower()

        # Remove extra whitespace
        text = ' '.join(text.split())

        # Remove punctuation
        import string
        text = text.translate(str.maketrans('', '', string.punctuation))

        return text

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        text = self._normalize_text(text)
        return text.split()

    def _jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """Calculate Jaccard similarity between two sets"""
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def get_statistics(self, result: DeduplicationResult) -> Dict:
        """Get statistics from deduplication result"""
        return {
            "total_examples": result.total_examples,
            "unique_examples": result.unique_examples,
            "duplicate_count": result.duplicate_count,
            "duplication_rate": round(result.duplication_rate * 100, 2),
            "duplicate_groups": len(result.duplicate_groups),
            "method": result.method,
            "similarity_threshold": result.similarity_threshold,
            "largest_duplicate_group": max(
                [len(group) for group in result.duplicate_groups],
                default=0
            )
        }

    def create_report(self, result: DeduplicationResult) -> str:
        """Create a text report from deduplication result"""
        stats = self.get_statistics(result)

        report = f"""Deduplication Report
{'=' * 50}

Method: {stats['method']}
Similarity Threshold: {stats['similarity_threshold']}

Total Examples: {stats['total_examples']}
Unique Examples: {stats['unique_examples']}
Duplicate Count: {stats['duplicate_count']}
Duplication Rate: {stats['duplication_rate']}%

Duplicate Groups: {stats['duplicate_groups']}
Largest Group: {stats['largest_duplicate_group']} examples

Recommendations:
"""

        if stats['duplication_rate'] > 20:
            report += "  - High duplication rate! Strongly recommend removing duplicates.\n"
        elif stats['duplication_rate'] > 10:
            report += "  - Moderate duplication detected. Consider deduplication.\n"
        elif stats['duplication_rate'] > 5:
            report += "  - Low duplication rate. Optional cleanup.\n"
        else:
            report += "  - Minimal duplication. Dataset is clean.\n"

        if stats['method'] == 'exact_hash':
            report += "  - Try fuzzy or semantic deduplication for better results.\n"

        return report
