"""
Version Manager

Manages dataset versions similar to Git
"""

import json
import hashlib
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path


@dataclass
class DatasetVersion:
    """Represents a dataset version"""
    version_id: str
    dataset_id: int
    version_number: int
    commit_message: str
    author: str
    timestamp: str
    parent_version: Optional[str]
    examples_hash: str
    examples_count: int
    metadata: Dict
    tags: List[str]


class VersionManager:
    """Manager for dataset versions"""

    def __init__(self, versions_dir: str = "./data/versions"):
        """
        Initialize version manager

        Args:
            versions_dir: Directory to store version data
        """
        self.versions_dir = Path(versions_dir)
        self.versions_dir.mkdir(parents=True, exist_ok=True)

    def create_version(
        self,
        dataset_id: int,
        examples: List[Dict],
        commit_message: str,
        author: str = "system",
        parent_version: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> DatasetVersion:
        """
        Create a new version of a dataset

        Args:
            dataset_id: ID of the dataset
            examples: List of examples in this version
            commit_message: Commit message describing changes
            author: Author of this version
            parent_version: ID of parent version (if any)
            tags: Optional tags for this version

        Returns:
            DatasetVersion object
        """
        # Get version number
        version_number = self._get_next_version_number(dataset_id)

        # Calculate examples hash
        examples_hash = self._hash_examples(examples)

        # Generate version ID
        version_id = self._generate_version_id(dataset_id, version_number)

        # Create version object
        version = DatasetVersion(
            version_id=version_id,
            dataset_id=dataset_id,
            version_number=version_number,
            commit_message=commit_message,
            author=author,
            timestamp=datetime.now().isoformat(),
            parent_version=parent_version,
            examples_hash=examples_hash,
            examples_count=len(examples),
            metadata={
                "created_at": datetime.now().isoformat()
            },
            tags=tags or []
        )

        # Save version data
        self._save_version(version, examples)

        return version

    def get_version(self, version_id: str) -> Optional[DatasetVersion]:
        """
        Get a specific version

        Args:
            version_id: Version ID

        Returns:
            DatasetVersion or None if not found
        """
        version_file = self.versions_dir / f"{version_id}.json"

        if not version_file.exists():
            return None

        with open(version_file, 'r') as f:
            data = json.load(f)

        return DatasetVersion(**data['version'])

    def get_version_examples(self, version_id: str) -> Optional[List[Dict]]:
        """
        Get examples from a specific version

        Args:
            version_id: Version ID

        Returns:
            List of examples or None if not found
        """
        examples_file = self.versions_dir / f"{version_id}_examples.jsonl"

        if not examples_file.exists():
            return None

        examples = []
        with open(examples_file, 'r') as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))

        return examples

    def list_versions(
        self,
        dataset_id: int,
        limit: Optional[int] = None
    ) -> List[DatasetVersion]:
        """
        List all versions for a dataset

        Args:
            dataset_id: Dataset ID
            limit: Optional limit on number of versions

        Returns:
            List of DatasetVersion objects
        """
        versions = []

        # Scan version files
        for version_file in self.versions_dir.glob(f"v{dataset_id}_*.json"):
            try:
                with open(version_file, 'r') as f:
                    data = json.load(f)
                    versions.append(DatasetVersion(**data['version']))
            except Exception as e:
                print(f"Error loading version {version_file}: {e}")
                continue

        # Sort by version number (descending)
        versions.sort(key=lambda v: v.version_number, reverse=True)

        if limit:
            versions = versions[:limit]

        return versions

    def get_version_history(
        self,
        dataset_id: int
    ) -> List[Dict]:
        """
        Get version history with change summaries

        Args:
            dataset_id: Dataset ID

        Returns:
            List of version history entries
        """
        versions = self.list_versions(dataset_id)
        history = []

        for i, version in enumerate(versions):
            entry = {
                "version_id": version.version_id,
                "version_number": version.version_number,
                "commit_message": version.commit_message,
                "author": version.author,
                "timestamp": version.timestamp,
                "examples_count": version.examples_count,
                "tags": version.tags
            }

            # Calculate changes from previous version
            if i < len(versions) - 1:
                prev_version = versions[i + 1]
                entry["changes"] = {
                    "examples_added": version.examples_count - prev_version.examples_count,
                    "is_same_content": version.examples_hash == prev_version.examples_hash
                }

            history.append(entry)

        return history

    def tag_version(
        self,
        version_id: str,
        tag: str
    ) -> bool:
        """
        Add a tag to a version

        Args:
            version_id: Version ID
            tag: Tag to add

        Returns:
            True if successful
        """
        version = self.get_version(version_id)

        if not version:
            return False

        if tag not in version.tags:
            version.tags.append(tag)
            self._update_version_metadata(version)

        return True

    def get_tagged_versions(
        self,
        dataset_id: int,
        tag: str
    ) -> List[DatasetVersion]:
        """
        Get all versions with a specific tag

        Args:
            dataset_id: Dataset ID
            tag: Tag to filter by

        Returns:
            List of DatasetVersion objects
        """
        versions = self.list_versions(dataset_id)
        return [v for v in versions if tag in v.tags]

    def rollback_to_version(
        self,
        version_id: str
    ) -> Optional[List[Dict]]:
        """
        Rollback to a specific version

        Args:
            version_id: Version ID to rollback to

        Returns:
            Examples from that version or None if not found
        """
        return self.get_version_examples(version_id)

    def delete_version(self, version_id: str) -> bool:
        """
        Delete a version

        Args:
            version_id: Version ID to delete

        Returns:
            True if deleted successfully
        """
        version_file = self.versions_dir / f"{version_id}.json"
        examples_file = self.versions_dir / f"{version_id}_examples.jsonl"

        try:
            if version_file.exists():
                version_file.unlink()
            if examples_file.exists():
                examples_file.unlink()
            return True
        except Exception as e:
            print(f"Error deleting version: {e}")
            return False

    def compare_versions(
        self,
        version_id1: str,
        version_id2: str
    ) -> Dict:
        """
        Compare two versions

        Args:
            version_id1: First version ID
            version_id2: Second version ID

        Returns:
            Comparison results
        """
        version1 = self.get_version(version_id1)
        version2 = self.get_version(version_id2)

        if not version1 or not version2:
            return {"error": "One or both versions not found"}

        return {
            "version1": {
                "version_id": version1.version_id,
                "version_number": version1.version_number,
                "timestamp": version1.timestamp,
                "examples_count": version1.examples_count
            },
            "version2": {
                "version_id": version2.version_id,
                "version_number": version2.version_number,
                "timestamp": version2.timestamp,
                "examples_count": version2.examples_count
            },
            "differences": {
                "examples_count_diff": version2.examples_count - version1.examples_count,
                "is_same_content": version1.examples_hash == version2.examples_hash,
                "time_between": self._calculate_time_diff(version1.timestamp, version2.timestamp)
            }
        }

    def _get_next_version_number(self, dataset_id: int) -> int:
        """Get the next version number for a dataset"""
        versions = self.list_versions(dataset_id)

        if not versions:
            return 1

        return max(v.version_number for v in versions) + 1

    def _generate_version_id(self, dataset_id: int, version_number: int) -> str:
        """Generate a version ID"""
        return f"v{dataset_id}_{version_number}"

    def _hash_examples(self, examples: List[Dict]) -> str:
        """Calculate hash of examples for change detection"""
        # Sort by ID if present to ensure consistent hashing
        examples_sorted = sorted(
            examples,
            key=lambda x: x.get('id', 0)
        )

        # Create JSON string
        examples_json = json.dumps(examples_sorted, sort_keys=True)

        # Hash it
        return hashlib.sha256(examples_json.encode()).hexdigest()

    def _save_version(self, version: DatasetVersion, examples: List[Dict]):
        """Save version data to disk"""
        # Save version metadata
        version_file = self.versions_dir / f"{version.version_id}.json"

        with open(version_file, 'w') as f:
            json.dump({
                "version": asdict(version)
            }, f, indent=2)

        # Save examples
        examples_file = self.versions_dir / f"{version.version_id}_examples.jsonl"

        with open(examples_file, 'w') as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')

    def _update_version_metadata(self, version: DatasetVersion):
        """Update version metadata file"""
        version_file = self.versions_dir / f"{version.version_id}.json"

        with open(version_file, 'w') as f:
            json.dump({
                "version": asdict(version)
            }, f, indent=2)

    def _calculate_time_diff(self, time1: str, time2: str) -> str:
        """Calculate time difference between two timestamps"""
        try:
            dt1 = datetime.fromisoformat(time1)
            dt2 = datetime.fromisoformat(time2)
            diff = abs((dt2 - dt1).total_seconds())

            if diff < 60:
                return f"{int(diff)} seconds"
            elif diff < 3600:
                return f"{int(diff / 60)} minutes"
            elif diff < 86400:
                return f"{int(diff / 3600)} hours"
            else:
                return f"{int(diff / 86400)} days"
        except:
            return "unknown"

    def get_statistics(self, dataset_id: int) -> Dict:
        """
        Get version statistics for a dataset

        Args:
            dataset_id: Dataset ID

        Returns:
            Statistics dictionary
        """
        versions = self.list_versions(dataset_id)

        if not versions:
            return {
                "total_versions": 0,
                "first_version": None,
                "latest_version": None
            }

        # Sort by version number
        versions_sorted = sorted(versions, key=lambda v: v.version_number)

        first_version = versions_sorted[0]
        latest_version = versions_sorted[-1]

        # Calculate total changes
        total_added = 0
        for i in range(1, len(versions_sorted)):
            diff = versions_sorted[i].examples_count - versions_sorted[i-1].examples_count
            if diff > 0:
                total_added += diff

        return {
            "total_versions": len(versions),
            "first_version": {
                "version_id": first_version.version_id,
                "timestamp": first_version.timestamp,
                "examples_count": first_version.examples_count
            },
            "latest_version": {
                "version_id": latest_version.version_id,
                "timestamp": latest_version.timestamp,
                "examples_count": latest_version.examples_count
            },
            "total_examples_added": total_added,
            "all_tags": list(set(tag for v in versions for tag in v.tags))
        }
