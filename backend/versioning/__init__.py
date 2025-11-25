"""
Dataset Versioning System

Git-like version control for datasets
"""

from .version_manager import VersionManager, DatasetVersion
from .diff import DatasetDiff, DiffResult
from .merge import DatasetMerger, MergeResult

__all__ = [
    "VersionManager",
    "DatasetVersion",
    "DatasetDiff",
    "DiffResult",
    "DatasetMerger",
    "MergeResult"
]
