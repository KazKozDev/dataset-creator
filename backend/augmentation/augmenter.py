"""
Data Augmentation Core

Main augmentation orchestrator and configuration
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import random


@dataclass
class AugmentationConfig:
    """Configuration for augmentation"""
    techniques: List[str] = field(default_factory=lambda: ["synonym"])
    samples_per_example: int = 1
    preserve_labels: bool = True
    seed: Optional[int] = None

    # Synonym replacement config
    synonym_ratio: float = 0.3  # % of words to replace

    # Random augmentation config
    random_swap_ratio: float = 0.1
    random_delete_ratio: float = 0.1
    random_insert_ratio: float = 0.1

    # Back-translation config
    target_languages: List[str] = field(default_factory=lambda: ["es", "fr", "de"])

    # Paraphrase config
    paraphrase_diversity: float = 0.7  # 0.0 = conservative, 1.0 = creative


@dataclass
class AugmentationResult:
    """Result of augmentation operation"""
    original_count: int
    augmented_count: int
    total_count: int
    technique_stats: Dict[str, int]
    examples: List[Dict]
    timestamp: str
    config: AugmentationConfig


class DataAugmenter:
    """Main data augmentation orchestrator"""

    def __init__(self, config: Optional[AugmentationConfig] = None):
        """
        Initialize augmenter

        Args:
            config: Augmentation configuration
        """
        self.config = config or AugmentationConfig()

        if self.config.seed is not None:
            random.seed(self.config.seed)

        # Import augmenters lazily to avoid circular imports
        from .techniques import (
            SynonymAugmenter,
            RandomAugmenter,
            ParaphraseAugmenter,
            BackTranslationAugmenter
        )

        # Initialize technique handlers
        self.augmenters = {
            "synonym": SynonymAugmenter(self.config),
            "random": RandomAugmenter(self.config),
            "paraphrase": ParaphraseAugmenter(self.config),
            "backtranslation": BackTranslationAugmenter(self.config)
        }

    def augment(
        self,
        examples: List[Dict],
        text_field: str = "text",
        label_field: Optional[str] = None
    ) -> AugmentationResult:
        """
        Augment a dataset

        Args:
            examples: Original examples
            text_field: Field containing text to augment
            label_field: Field containing labels to preserve

        Returns:
            AugmentationResult with augmented examples
        """
        augmented_examples = []
        technique_stats = {tech: 0 for tech in self.config.techniques}

        for example in examples:
            # Keep original
            augmented_examples.append(example.copy())

            # Generate augmented versions
            for _ in range(self.config.samples_per_example):
                # Randomly select technique
                technique = random.choice(self.config.techniques)

                # Apply augmentation
                augmenter = self.augmenters.get(technique)
                if not augmenter:
                    continue

                augmented_text = augmenter.augment_text(
                    example.get(text_field, "")
                )

                if augmented_text and augmented_text != example.get(text_field):
                    # Create augmented example
                    aug_example = example.copy()
                    aug_example[text_field] = augmented_text

                    # Add metadata
                    aug_example["_augmented"] = True
                    aug_example["_augmentation_technique"] = technique
                    aug_example["_original_id"] = example.get("id", "unknown")

                    augmented_examples.append(aug_example)
                    technique_stats[technique] += 1

        return AugmentationResult(
            original_count=len(examples),
            augmented_count=len(augmented_examples) - len(examples),
            total_count=len(augmented_examples),
            technique_stats=technique_stats,
            examples=augmented_examples,
            timestamp=datetime.now().isoformat(),
            config=self.config
        )

    def augment_text(
        self,
        text: str,
        technique: Optional[str] = None
    ) -> str:
        """
        Augment a single text

        Args:
            text: Text to augment
            technique: Specific technique to use (random if None)

        Returns:
            Augmented text
        """
        if not technique:
            technique = random.choice(self.config.techniques)

        augmenter = self.augmenters.get(technique)
        if not augmenter:
            return text

        return augmenter.augment_text(text)

    def augment_batch(
        self,
        texts: List[str],
        technique: Optional[str] = None
    ) -> List[str]:
        """
        Augment a batch of texts

        Args:
            texts: Texts to augment
            technique: Specific technique to use

        Returns:
            List of augmented texts
        """
        return [
            self.augment_text(text, technique)
            for text in texts
        ]

    def get_available_techniques(self) -> List[str]:
        """Get list of available augmentation techniques"""
        return list(self.augmenters.keys())

    def get_statistics(self, result: AugmentationResult) -> Dict:
        """
        Get augmentation statistics

        Args:
            result: Augmentation result

        Returns:
            Statistics dictionary
        """
        total_augmented = sum(result.technique_stats.values())

        return {
            "original_count": result.original_count,
            "augmented_count": result.augmented_count,
            "total_count": result.total_count,
            "augmentation_ratio": round(result.augmented_count / result.original_count, 2) if result.original_count > 0 else 0,
            "technique_breakdown": result.technique_stats,
            "technique_percentages": {
                tech: round(count / total_augmented * 100, 1) if total_augmented > 0 else 0
                for tech, count in result.technique_stats.items()
            },
            "timestamp": result.timestamp
        }
