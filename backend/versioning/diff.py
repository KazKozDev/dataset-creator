"""
Dataset Diff

Calculate differences between dataset versions
"""

import hashlib
import json
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass


@dataclass
class DiffResult:
    """Result of diff operation"""
    added: List[Dict]
    removed: List[Dict]
    modified: List[Tuple[Dict, Dict]]  # (old, new)
    unchanged: int
    total_changes: int
    summary: Dict


class DatasetDiff:
    """Calculator for dataset differences"""

    def diff(
        self,
        examples1: List[Dict],
        examples2: List[Dict],
        id_field: str = "id"
    ) -> DiffResult:
        """
        Calculate diff between two sets of examples

        Args:
            examples1: First set of examples (old)
            examples2: Second set of examples (new)
            id_field: Field to use as unique identifier

        Returns:
            DiffResult with changes
        """
        # Create ID-to-example mappings
        map1 = self._create_id_map(examples1, id_field)
        map2 = self._create_id_map(examples2, id_field)

        # Find added, removed, and potentially modified
        ids1 = set(map1.keys())
        ids2 = set(map2.keys())

        added_ids = ids2 - ids1
        removed_ids = ids1 - ids2
        common_ids = ids1 & ids2

        # Extract added and removed examples
        added = [map2[id] for id in added_ids]
        removed = [map1[id] for id in removed_ids]

        # Check for modifications in common examples
        modified = []
        unchanged = 0

        for id in common_ids:
            ex1 = map1[id]
            ex2 = map2[id]

            if self._examples_differ(ex1, ex2):
                modified.append((ex1, ex2))
            else:
                unchanged += 1

        # Calculate summary
        total_changes = len(added) + len(removed) + len(modified)

        summary = {
            "total_examples_before": len(examples1),
            "total_examples_after": len(examples2),
            "added_count": len(added),
            "removed_count": len(removed),
            "modified_count": len(modified),
            "unchanged_count": unchanged,
            "net_change": len(examples2) - len(examples1)
        }

        return DiffResult(
            added=added,
            removed=removed,
            modified=modified,
            unchanged=unchanged,
            total_changes=total_changes,
            summary=summary
        )

    def diff_by_content(
        self,
        examples1: List[Dict],
        examples2: List[Dict],
        text_field: str = "text"
    ) -> DiffResult:
        """
        Calculate diff based on content rather than IDs

        Args:
            examples1: First set of examples (old)
            examples2: Second set of examples (new)
            text_field: Field containing text content

        Returns:
            DiffResult with changes
        """
        # Create content hash to example mappings
        map1 = self._create_content_map(examples1, text_field)
        map2 = self._create_content_map(examples2, text_field)

        # Find added and removed
        hashes1 = set(map1.keys())
        hashes2 = set(map2.keys())

        added_hashes = hashes2 - hashes1
        removed_hashes = hashes1 - hashes2
        common_hashes = hashes1 & hashes2

        # Extract examples
        added = [map2[h] for h in added_hashes]
        removed = [map1[h] for h in removed_hashes]
        unchanged = len(common_hashes)

        # No modifications in content-based diff
        modified = []

        total_changes = len(added) + len(removed)

        summary = {
            "total_examples_before": len(examples1),
            "total_examples_after": len(examples2),
            "added_count": len(added),
            "removed_count": len(removed),
            "modified_count": 0,
            "unchanged_count": unchanged,
            "net_change": len(examples2) - len(examples1)
        }

        return DiffResult(
            added=added,
            removed=removed,
            modified=modified,
            unchanged=unchanged,
            total_changes=total_changes,
            summary=summary
        )

    def create_patch(
        self,
        diff_result: DiffResult,
        id_field: str = "id"
    ) -> Dict:
        """
        Create a patch file from diff result

        Args:
            diff_result: Diff result to create patch from
            id_field: ID field name

        Returns:
            Patch dictionary
        """
        return {
            "patch_version": "1.0",
            "summary": diff_result.summary,
            "changes": {
                "added": diff_result.added,
                "removed": [ex[id_field] for ex in diff_result.removed],
                "modified": [
                    {
                        "id": old[id_field],
                        "old": old,
                        "new": new
                    }
                    for old, new in diff_result.modified
                ]
            }
        }

    def apply_patch(
        self,
        examples: List[Dict],
        patch: Dict,
        id_field: str = "id"
    ) -> List[Dict]:
        """
        Apply a patch to a set of examples

        Args:
            examples: Original examples
            patch: Patch to apply
            id_field: ID field name

        Returns:
            Modified examples
        """
        # Create mutable copy
        result = list(examples)

        # Create ID map for quick lookup
        id_map = {ex[id_field]: i for i, ex in enumerate(result)}

        # Apply removals
        for removed_id in patch["changes"]["removed"]:
            if removed_id in id_map:
                idx = id_map[removed_id]
                result[idx] = None  # Mark for removal

        # Remove None entries
        result = [ex for ex in result if ex is not None]

        # Apply modifications
        id_map = {ex[id_field]: i for i, ex in enumerate(result)}
        for mod in patch["changes"]["modified"]:
            mod_id = mod["id"]
            if mod_id in id_map:
                idx = id_map[mod_id]
                result[idx] = mod["new"]

        # Apply additions
        result.extend(patch["changes"]["added"])

        return result

    def _create_id_map(self, examples: List[Dict], id_field: str) -> Dict:
        """Create mapping from ID to example"""
        id_map = {}

        for ex in examples:
            if id_field in ex:
                id_map[ex[id_field]] = ex
            else:
                # Generate hash-based ID if no ID field
                ex_hash = self._hash_example(ex)
                id_map[ex_hash] = ex

        return id_map

    def _create_content_map(self, examples: List[Dict], text_field: str) -> Dict:
        """Create mapping from content hash to example"""
        content_map = {}

        for ex in examples:
            text = self._extract_text(ex, text_field)
            content_hash = self._hash_text(text)
            content_map[content_hash] = ex

        return content_map

    def _extract_text(self, example: Dict, text_field: str) -> str:
        """Extract text from example"""
        if text_field in example:
            return str(example[text_field])

        for field in ["text", "content", "prompt"]:
            if field in example:
                return str(example[field])

        return ""

    def _hash_example(self, example: Dict) -> str:
        """Hash an example for comparison"""
        example_json = json.dumps(example, sort_keys=True)
        return hashlib.md5(example_json.encode()).hexdigest()

    def _hash_text(self, text: str) -> str:
        """Hash text content"""
        return hashlib.md5(text.encode()).hexdigest()

    def _examples_differ(self, ex1: Dict, ex2: Dict) -> bool:
        """Check if two examples are different"""
        hash1 = self._hash_example(ex1)
        hash2 = self._hash_example(ex2)
        return hash1 != hash2

    def create_diff_report(self, diff_result: DiffResult) -> str:
        """
        Create a human-readable diff report

        Args:
            diff_result: Diff result

        Returns:
            Formatted report string
        """
        report = f"""Dataset Diff Report
{'=' * 50}

Summary:
  Examples Before: {diff_result.summary['total_examples_before']}
  Examples After:  {diff_result.summary['total_examples_after']}
  Net Change:      {diff_result.summary['net_change']:+d}

Changes:
  Added:           {diff_result.summary['added_count']}
  Removed:         {diff_result.summary['removed_count']}
  Modified:        {diff_result.summary['modified_count']}
  Unchanged:       {diff_result.summary['unchanged_count']}

Total Changes:     {diff_result.total_changes}
"""

        if diff_result.added:
            report += f"\nAdded Examples ({len(diff_result.added)}):\n"
            for i, ex in enumerate(diff_result.added[:5], 1):
                report += f"  {i}. {self._truncate_example(ex)}\n"
            if len(diff_result.added) > 5:
                report += f"  ... and {len(diff_result.added) - 5} more\n"

        if diff_result.removed:
            report += f"\nRemoved Examples ({len(diff_result.removed)}):\n"
            for i, ex in enumerate(diff_result.removed[:5], 1):
                report += f"  {i}. {self._truncate_example(ex)}\n"
            if len(diff_result.removed) > 5:
                report += f"  ... and {len(diff_result.removed) - 5} more\n"

        if diff_result.modified:
            report += f"\nModified Examples ({len(diff_result.modified)}):\n"
            for i, (old, new) in enumerate(diff_result.modified[:3], 1):
                report += f"  {i}. {self._truncate_example(old)} → {self._truncate_example(new)}\n"
            if len(diff_result.modified) > 3:
                report += f"  ... and {len(diff_result.modified) - 3} more\n"

        return report

    def _truncate_example(self, example: Dict, max_len: int = 60) -> str:
        """Create a short string representation of an example"""
        # Try to find text content
        text = ""
        for field in ["text", "content", "prompt", "instruction"]:
            if field in example:
                text = str(example[field])
                break

        if not text:
            text = str(example)

        if len(text) > max_len:
            text = text[:max_len] + "..."

        return text
