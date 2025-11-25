"""
Dataset Merger

Merge multiple dataset versions
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import hashlib
import json


@dataclass
class MergeResult:
    """Result of merge operation"""
    merged_examples: List[Dict]
    conflicts: List[Dict]
    stats: Dict
    success: bool


class DatasetMerger:
    """Merger for dataset versions"""

    def merge(
        self,
        base_examples: List[Dict],
        branch1_examples: List[Dict],
        branch2_examples: List[Dict],
        strategy: str = "union",
        id_field: str = "id"
    ) -> MergeResult:
        """
        Merge two dataset branches

        Args:
            base_examples: Common ancestor examples
            branch1_examples: First branch examples
            branch2_examples: Second branch examples
            strategy: Merge strategy - "union", "intersection", "prefer_branch1", "prefer_branch2"
            id_field: Field to use as unique identifier

        Returns:
            MergeResult with merged data
        """
        if strategy == "union":
            return self._merge_union(branch1_examples, branch2_examples, id_field)
        elif strategy == "intersection":
            return self._merge_intersection(branch1_examples, branch2_examples, id_field)
        elif strategy == "prefer_branch1":
            return self._merge_prefer(branch1_examples, branch2_examples, id_field, prefer_first=True)
        elif strategy == "prefer_branch2":
            return self._merge_prefer(branch1_examples, branch2_examples, id_field, prefer_first=False)
        else:
            raise ValueError(f"Unknown merge strategy: {strategy}")

    def _merge_union(
        self,
        examples1: List[Dict],
        examples2: List[Dict],
        id_field: str
    ) -> MergeResult:
        """
        Union merge: include all examples from both sets

        Conflicts occur when same ID has different content
        """
        # Create ID maps
        map1 = self._create_id_map(examples1, id_field)
        map2 = self._create_id_map(examples2, id_field)

        # Find IDs
        ids1 = set(map1.keys())
        ids2 = set(map2.keys())

        # Unique to each branch
        unique_to_1 = ids1 - ids2
        unique_to_2 = ids2 - ids1
        common_ids = ids1 & ids2

        merged = []
        conflicts = []

        # Add unique examples from both
        for id in unique_to_1:
            merged.append(map1[id])

        for id in unique_to_2:
            merged.append(map2[id])

        # Handle common IDs
        for id in common_ids:
            ex1 = map1[id]
            ex2 = map2[id]

            if self._examples_equal(ex1, ex2):
                # Same content, no conflict
                merged.append(ex1)
            else:
                # Conflict: different content for same ID
                conflicts.append({
                    "id": id,
                    "branch1": ex1,
                    "branch2": ex2,
                    "reason": "content_differs"
                })

                # For union, include both (with modified IDs)
                merged.append(ex1)

        stats = {
            "total_examples": len(merged),
            "from_branch1": len(unique_to_1),
            "from_branch2": len(unique_to_2),
            "common": len(common_ids),
            "conflicts": len(conflicts)
        }

        return MergeResult(
            merged_examples=merged,
            conflicts=conflicts,
            stats=stats,
            success=len(conflicts) == 0
        )

    def _merge_intersection(
        self,
        examples1: List[Dict],
        examples2: List[Dict],
        id_field: str
    ) -> MergeResult:
        """
        Intersection merge: only include examples present in both sets
        """
        # Create ID maps
        map1 = self._create_id_map(examples1, id_field)
        map2 = self._create_id_map(examples2, id_field)

        # Find common IDs
        common_ids = set(map1.keys()) & set(map2.keys())

        merged = []
        conflicts = []

        for id in common_ids:
            ex1 = map1[id]
            ex2 = map2[id]

            if self._examples_equal(ex1, ex2):
                merged.append(ex1)
            else:
                # Conflict: keep one (prefer first)
                conflicts.append({
                    "id": id,
                    "branch1": ex1,
                    "branch2": ex2,
                    "reason": "content_differs",
                    "resolution": "kept_branch1"
                })
                merged.append(ex1)

        stats = {
            "total_examples": len(merged),
            "common": len(common_ids),
            "conflicts": len(conflicts)
        }

        return MergeResult(
            merged_examples=merged,
            conflicts=conflicts,
            stats=stats,
            success=len(conflicts) == 0
        )

    def _merge_prefer(
        self,
        examples1: List[Dict],
        examples2: List[Dict],
        id_field: str,
        prefer_first: bool
    ) -> MergeResult:
        """
        Preference merge: prefer one branch over the other
        """
        # Create ID maps
        map1 = self._create_id_map(examples1, id_field)
        map2 = self._create_id_map(examples2, id_field)

        # Find IDs
        ids1 = set(map1.keys())
        ids2 = set(map2.keys())
        all_ids = ids1 | ids2

        merged = []
        conflicts = []

        for id in all_ids:
            has_in_1 = id in ids1
            has_in_2 = id in ids2

            if has_in_1 and has_in_2:
                ex1 = map1[id]
                ex2 = map2[id]

                if not self._examples_equal(ex1, ex2):
                    # Conflict, but we have preference
                    conflicts.append({
                        "id": id,
                        "branch1": ex1,
                        "branch2": ex2,
                        "reason": "content_differs",
                        "resolution": "preferred_branch1" if prefer_first else "preferred_branch2"
                    })

                # Prefer one
                merged.append(ex1 if prefer_first else ex2)
            elif has_in_1:
                merged.append(map1[id])
            else:
                merged.append(map2[id])

        stats = {
            "total_examples": len(merged),
            "from_branch1": len(ids1 - ids2),
            "from_branch2": len(ids2 - ids1),
            "common": len(ids1 & ids2),
            "conflicts": len(conflicts),
            "preferred": "branch1" if prefer_first else "branch2"
        }

        return MergeResult(
            merged_examples=merged,
            conflicts=conflicts,
            stats=stats,
            success=True  # Always succeeds with preference
        )

    def merge_by_content(
        self,
        examples1: List[Dict],
        examples2: List[Dict],
        text_field: str = "text"
    ) -> MergeResult:
        """
        Merge by content (deduplicate based on content hash)

        Args:
            examples1: First set of examples
            examples2: Second set of examples
            text_field: Field containing text content

        Returns:
            MergeResult with deduplicated examples
        """
        # Create content hash maps
        map1 = self._create_content_map(examples1, text_field)
        map2 = self._create_content_map(examples2, text_field)

        # Union of all content
        all_hashes = set(map1.keys()) | set(map2.keys())

        merged = []
        for hash_val in all_hashes:
            if hash_val in map1:
                merged.append(map1[hash_val])
            else:
                merged.append(map2[hash_val])

        stats = {
            "total_examples": len(merged),
            "from_branch1_only": len(set(map1.keys()) - set(map2.keys())),
            "from_branch2_only": len(set(map2.keys()) - set(map1.keys())),
            "common": len(set(map1.keys()) & set(map2.keys())),
            "conflicts": 0
        }

        return MergeResult(
            merged_examples=merged,
            conflicts=[],
            stats=stats,
            success=True
        )

    def resolve_conflicts(
        self,
        merge_result: MergeResult,
        resolution_strategy: str = "keep_both"
    ) -> List[Dict]:
        """
        Resolve merge conflicts

        Args:
            merge_result: Merge result with conflicts
            resolution_strategy: How to resolve - "keep_both", "keep_first", "keep_second"

        Returns:
            Resolved examples
        """
        if not merge_result.conflicts:
            return merge_result.merged_examples

        resolved = list(merge_result.merged_examples)

        for conflict in merge_result.conflicts:
            if resolution_strategy == "keep_both":
                # Already in merged_examples usually
                pass
            elif resolution_strategy == "keep_first":
                # Find and replace with branch1 version
                resolved = [
                    conflict["branch1"] if self._examples_equal(ex, conflict["branch2"]) else ex
                    for ex in resolved
                ]
            elif resolution_strategy == "keep_second":
                # Find and replace with branch2 version
                resolved = [
                    conflict["branch2"] if self._examples_equal(ex, conflict["branch1"]) else ex
                    for ex in resolved
                ]

        return resolved

    def _create_id_map(self, examples: List[Dict], id_field: str) -> Dict:
        """Create mapping from ID to example"""
        id_map = {}

        for ex in examples:
            if id_field in ex:
                id_map[ex[id_field]] = ex
            else:
                # Generate hash-based ID
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
        """Hash an example"""
        example_json = json.dumps(example, sort_keys=True)
        return hashlib.md5(example_json.encode()).hexdigest()

    def _hash_text(self, text: str) -> str:
        """Hash text content"""
        return hashlib.md5(text.encode()).hexdigest()

    def _examples_equal(self, ex1: Dict, ex2: Dict) -> bool:
        """Check if two examples are equal"""
        hash1 = self._hash_example(ex1)
        hash2 = self._hash_example(ex2)
        return hash1 == hash2

    def create_merge_report(self, merge_result: MergeResult) -> str:
        """
        Create a human-readable merge report

        Args:
            merge_result: Merge result

        Returns:
            Formatted report string
        """
        report = f"""Dataset Merge Report
{'=' * 50}

Status: {'✓ SUCCESS' if merge_result.success else '⚠ CONFLICTS DETECTED'}

Statistics:
  Total Merged Examples: {merge_result.stats['total_examples']}
"""

        if 'from_branch1' in merge_result.stats:
            report += f"  From Branch 1 Only:   {merge_result.stats['from_branch1']}\n"
        if 'from_branch2' in merge_result.stats:
            report += f"  From Branch 2 Only:   {merge_result.stats['from_branch2']}\n"
        if 'common' in merge_result.stats:
            report += f"  Common:                {merge_result.stats['common']}\n"

        report += f"  Conflicts:             {merge_result.stats['conflicts']}\n"

        if merge_result.conflicts:
            report += f"\nConflicts ({len(merge_result.conflicts)}):\n"
            for i, conflict in enumerate(merge_result.conflicts[:5], 1):
                report += f"  {i}. ID: {conflict['id']} - {conflict['reason']}\n"
                if 'resolution' in conflict:
                    report += f"     Resolution: {conflict['resolution']}\n"

            if len(merge_result.conflicts) > 5:
                report += f"  ... and {len(merge_result.conflicts) - 5} more conflicts\n"

            report += "\nRecommendation: Review and resolve conflicts before using merged dataset.\n"

        return report
