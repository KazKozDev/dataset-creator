"""
Data Augmentation System

Text augmentation techniques for dataset expansion
"""

from .augmenter import (
    DataAugmenter,
    AugmentationConfig,
    AugmentationResult
)
from .techniques import (
    ParaphraseAugmenter,
    SynonymAugmenter,
    BackTranslationAugmenter,
    RandomAugmenter
)

__all__ = [
    "DataAugmenter",
    "AugmentationConfig",
    "AugmentationResult",
    "ParaphraseAugmenter",
    "SynonymAugmenter",
    "BackTranslationAugmenter",
    "RandomAugmenter"
]
